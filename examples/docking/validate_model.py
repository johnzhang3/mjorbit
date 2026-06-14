# # Run this on your ISS.obj
# with open("examples/docking/ISS2.obj", 'r') as f:
#     lines = f.readlines()

# vertices = sum(1 for line in lines if line.startswith('v '))
# faces = sum(1 for line in lines if line.startswith('f '))
# normals = sum(1 for line in lines if line.startswith('vn '))

# print(f"Vertices: {vertices}")
# print(f"Normals: {normals}")
# print(f"Faces: {faces}")
# print(f"Total lines: {len(lines)}")

# if faces == 0:
#     print("\n⚠️  WARNING: No faces found! File may be corrupted or incompletely exported.")
# else:
#     print(f"\n✓ Mesh appears valid ({faces} triangles)")

import mujoco
import mujoco.viewer

# Minimal XML - no orbit plugin
xml = """<?xml version="1.0" ?>
<mujoco model="iss_test">
  <asset>
    <mesh name="iss" file="examples/docking/ISS.obj" scale="1 1 1" />
  </asset>

  <worldbody>
    <!-- Test box (sanity check) -->
    <geom type="box" size="1 1 1" rgba="1 0 0 1" pos="0 0 0" />
    
    <!-- ISS mesh -->
    <body name="iss">
      <geom type="mesh" mesh="iss" rgba="0.8 0.8 0.8 1" />
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(xml)
    print("✓ Model loaded")
    print(f"  Geoms: {model.ngeom}")
    print(f"  Meshes: {model.nmesh}")
    
    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()