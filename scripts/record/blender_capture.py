"""Blender (Cycles) render of the capture-and-stabilize runs exported by ``blender_export.py``.

The runs fly side by side above a textured Earth: each is offset along-track in the local
orbital frame (a display offset, not physics), the camera rides LVLH looking along the orbit
normal with radial up, and the sun is fixed in LVLH so the scene stays lit. Frames are written
as JPEGs; add labels and encode afterwards.

    blender -b -P scripts/record/blender_capture.py -- frames.npz out_dir \\
        --earth src/viewer/assets/earth_day_5400.jpg [--frames 1 30] [--samples 32]
"""

import argparse
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

WORLD = 1e-4                 # render units per metre for the Earth and orbit (spacecraft are 1:1)
R_EARTH_KM = 6378.137
OMEGA_EARTH = 7.2921159e-5   # rad/s


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out_dir")
    ap.add_argument("--earth", required=True, help="equirectangular Earth texture")
    ap.add_argument("--res", type=int, nargs=2, default=(1920, 1080))
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--frames-range", dest="frames_range", type=int, nargs=2, default=None,
                    metavar=("FIRST", "LAST"), help="render only these frames (1-based)")
    ap.add_argument("--frame-step", type=int, default=1, help="render every Nth frame (for tests)")
    ap.add_argument("--fov", type=float, default=30.0, help="vertical field of view (deg)")
    ap.add_argument("--distance", type=float, default=30.0, help="camera distance (m)")
    ap.add_argument("--offset", type=float, nargs=3, default=(0.3, 0.0, 1.0),
                    metavar=("RADIAL", "ALONG", "CROSS"), help="camera direction in LVLH")
    ap.add_argument("--spacing", type=float, default=8.0,
                    help="along-track display offset of each run from the centre (m)")
    return ap.parse_args(argv)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def material(name, rgb, metallic=0.0, roughness=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(rgb), 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return m


def mesh_object(name, build, smooth, mat):
    bm = bmesh.new()
    build(bm)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = smooth
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj.rotation_mode = "QUATERNION"
    return obj


def geom_builder(gtype, size):
    """bmesh builder for a MuJoCo geom (box 6, sphere 2, capsule 3, cylinder 5), local frame."""
    r, h = float(size[0]), float(size[1])

    def box(bm):
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)

    def sphere(bm):
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=r)

    def cylinder(bm):
        bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=r, radius2=r, depth=2 * h)

    def capsule(bm):
        cylinder(bm)
        for z in (h, -h):
            res = bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=r)
            bmesh.ops.translate(bm, vec=Vector((0, 0, z)), verts=res["verts"])

    return {6: (box, False), 2: (sphere, True), 3: (capsule, True), 5: (cylinder, True)}[int(gtype)]


GEOM_LOOK = [  # (rgb or None for the model's, metallic, roughness), in export order:
    (None, 0.4, 0.35),              # bus
    ((0.82, 0.83, 0.85), 0.2, 0.4),  # link1
    ((0.82, 0.83, 0.85), 0.2, 0.4),  # link2
    (None, 1.0, 0.25),              # end effector
    (None, 0.1, 0.55),              # payload
    (None, 0.3, 0.35),              # panel
    (None, 0.3, 0.35),              # panel
]


def lvlh(R, V):
    r = R / np.linalg.norm(R)
    w = np.cross(R, V)
    w /= np.linalg.norm(w)
    return r, np.cross(w, r), w


def look_quat(loc, target, up):
    z = -(target - loc).normalized()
    x = up.cross(z).normalized()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed().to_quaternion()


def setup_render(args, n_frames, fps):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            break
        except TypeError:
            continue
    (prefs.refresh_devices if hasattr(prefs, "refresh_devices") else prefs.get_devices)()
    for d in prefs.devices:
        d.use = d.type != "CPU"
    sc.cycles.device = "GPU"
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.render.resolution_x, sc.render.resolution_y = args.res
    sc.render.resolution_percentage = 100
    sc.render.fps = int(fps)
    sc.frame_start, sc.frame_end = (args.frames_range or (1, n_frames))
    sc.frame_step = args.frame_step
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 93
    sc.render.filepath = args.out_dir.rstrip("/") + "/frame_####"
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass


def build_world():
    w = bpy.data.worlds.new("space")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1500.0, 1500.0, 1500.0)
    noise = nt.nodes.new("ShaderNodeTexWhiteNoise")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 0.9996
    ramp.color_ramp.elements[1].color = (0.8, 0.85, 1.0, 1)
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Value"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0


def build_earth(texture):
    radius = R_EARTH_KM * 1000.0 * WORLD
    mat = bpy.data.materials.new("earth")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(texture)
    tex.interpolation = "Cubic"
    gamma = nt.nodes.new("ShaderNodeGamma")
    gamma.inputs["Gamma"].default_value = 0.65
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 1.2
    nt.links.new(tex.outputs["Color"], gamma.inputs["Color"])
    nt.links.new(gamma.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8

    def sphere(bm):
        bm.loops.layers.uv.new("UVMap")
        bmesh.ops.create_uvsphere(bm, u_segments=1024, v_segments=512, radius=radius, calc_uvs=True)

    earth = mesh_object("earth", sphere, True, mat)
    earth.rotation_mode = "XYZ"

    atmo_mat = bpy.data.materials.new("atmosphere")
    atmo_mat.use_nodes = True
    nt = atmo_mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = nt.nodes["Material Output"]
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.2
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 7.0
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.25, 0.5, 1.0, 1)
    em.inputs["Strength"].default_value = 0.8
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    nt.links.new(pw.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    def shell(bm):
        bmesh.ops.create_uvsphere(bm, u_segments=512, v_segments=256, radius=radius * 1.012)

    atmo = mesh_object("atmosphere", shell, True, atmo_mat)
    atmo.parent = earth
    return earth


def main():
    args = parse()
    d = np.load(args.frames)
    pos, quat = d["pos"], d["quat"]            # (runs, F, G, 3|4)
    t, R_km, V = d["t"], d["R_km"], d["V_kms"]
    n_runs, F, G = pos.shape[:3]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render(args, F, float(d["fps"]))
    build_world()
    earth = build_earth(args.earth)

    def geom_object(k, g):
        build, smooth = geom_builder(d["geom_type"][g], d["geom_size"][g])
        rgb = GEOM_LOOK[g][0] or d["geom_rgba"][g][:3]
        return mesh_object(f"run{k}_geom{g}", build, smooth,
                           material(f"run{k}_mat{g}", rgb, *GEOM_LOOK[g][1:]))

    objs = [[geom_object(k, g) for g in range(G)] for k in range(n_runs)]
    # The spacecraft are drawn ~1e4 times oversized, so their shadows on the Earth would be too.
    for row in objs:
        for o in row:
            o.visible_shadow = False
    # display offsets: run 0 on the image left (+along-track), the next on the right
    sides = np.linspace(1.0, -1.0, n_runs) if n_runs > 1 else np.zeros(1)

    cam_data = bpy.data.cameras.new("camera")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = math.radians(args.fov)
    cam_data.clip_start, cam_data.clip_end = 0.1, 5000.0
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"

    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 5.5
    sun_data.angle = math.radians(0.53)
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_mode = "QUATERNION"

    off_dir = np.asarray(args.offset, dtype=float)
    off_dir /= np.linalg.norm(off_dir)
    first, last = args.frames_range or (1, F)
    for f in range(first - 1, last):
        frame = f + 1
        r, s, w = lvlh(R_km[f], V[f])
        centers = []
        for k in range(n_runs):
            shift = sides[k] * args.spacing * s
            for g in range(G):
                o = objs[k][g]
                o.location = Vector(pos[k, f, g] + shift)
                o.rotation_quaternion = Quaternion(quat[k, f, g])
                o.keyframe_insert("location", frame=frame)
                o.keyframe_insert("rotation_quaternion", frame=frame)
            centers.append(0.5 * (pos[k, f, 0] + pos[k, f, 4]) + shift)   # bus and payload
        target = Vector(np.mean(centers, axis=0))
        loc = target + Vector(args.distance * (off_dir[0] * r + off_dir[1] * s + off_dir[2] * w))
        cam.location = loc
        cam.rotation_quaternion = look_quat(loc, target, Vector(r))
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_quaternion", frame=frame)
        # sun from above, image left and the camera side
        light_from = Vector(0.55 * r + 0.35 * s + 0.75 * w).normalized()
        sun.rotation_quaternion = (-light_from).to_track_quat("-Z", "Y")
        sun.keyframe_insert("rotation_quaternion", frame=frame)
        earth.location = Vector(-R_km[f] * 1000.0 * WORLD)
        earth.rotation_euler = (0.0, 0.0, OMEGA_EARTH * float(t[f]))
        earth.keyframe_insert("location", frame=frame)
        earth.keyframe_insert("rotation_euler", frame=frame)

    print(f"scene built: {n_runs} runs x {G} geoms, {F} frames; rendering "
          f"{bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}", flush=True)
    bpy.ops.render.render(animation=True)


main()
