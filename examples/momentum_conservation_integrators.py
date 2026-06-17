"""Angular-momentum conservation vs MuJoCo integrator -- and the joint-limit trap.

A freely tumbling spacecraft with internal joints must conserve its total angular
momentum exactly (internal joint forces obey Newton's third law, external torques
are off here). We spin a bus + 2-link arm up to a known L0, let it tumble freely
(arm servo-held in a fixed pose, no external torque), and track the relative drift
|L(t) - L0| / |L0| for each of MuJoCo's integrators in TWO regimes:

  (A) joints free (no limits): the arm sits in a fixed bent pose while the whole
      body tumbles. Euler/implicit/implicitfast drift only mildly (~1.3% over 30 s)
      from the explicitly-integrated gyroscopic term, and RK4 conserves to ~1e-3%.
      This is the benign case -- and note implicit/implicitfast do NOT beat Euler
      for an articulated body, unlike the single-rigid-body case in issue #12.

  (B) a joint-limit constraint is VIOLATED (the held pose sits outside the joint
      range, so the limit constraint is active and penetrating): the first-order
      integrators (Euler/implicit/implicitfast) inject LARGE spurious angular
      momentum on the very first step -- ~7% here, and it grows without bound as
      the penetration deepens (>1000% when the joint is slammed far past its stop).
      RK4 stays clean (~0.1%).

So the dramatic "momentum jump after the first step" is an integrator x constraint
interaction: MuJoCo's first-order integrators resolve a violated joint-limit
constraint badly, while RK4 does not. It is vanilla MuJoCo (mjorbit inherits
mj_step; reproduced with raw mujoco.mj_step), and it matters because joint limits
are ubiquitous in articulated spacecraft. If your model can reach or exceed a
joint limit and you need a clean momentum signal (e.g. magnetorquer momentum
dumping), use RK4 -- implicit/implicitfast are NOT enough.

UNITS WARNING (how this bites by accident): without ``<compiler angle="radian"/>``
MuJoCo parses joint ``range`` in DEGREES, so ``range="-3.14 3.14"`` is +-3.14 deg
= +-0.055 rad. A radian-valued initial pose (e.g. 0.6 rad) then starts ~11x past
the limit -- silently putting you in regime (B). Always set the angle unit.

This is distinct from the orbit-position result in the mjorbit paper (Fig. 3),
where semi-implicit Euler in the orbit-following frame is excellent because the
orbit Hamiltonian is separable.

L is computed about the SYSTEM CoM in the orbit-following world frame (axes
parallel to ECI, non-rotating). That equals the absolute-ECI angular momentum
because L-about-CoM is invariant to the frame's uniform translation, and no
Coriolis/centrifugal term is needed since the axes do not rotate. MuJoCo ``cvel``
linear velocity is referenced at the subtree CoM, so we transport it to each
body's CoM with ``omega_i x (xipos_i - CoM)`` before forming L.

The integrator is set on ``model.opt.integrator`` *after* compilation rather than
in the XML on purpose: mjorbit's compile step upgrades an unspecified or explicit
``Euler`` integrator to ``implicitfast`` (issue #12 fix), which would silently turn
the "Euler" curve into an implicitfast curve. Setting it post-compile pins each
curve to exactly the integrator it is labelled with, on any branch.

See the accompanying issue for the open question of the precise MuJoCo mechanism
and its relation to the implicitfast default (issue #12).

Usage:
    pixi run -e report python examples/momentum_conservation_integrators.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH

# Free-floating bus + 2-link arm. All orbital perturbations and the gravity-
# gradient torque are OFF, so the only torques are internal -> total angular
# momentum must be conserved. {dt} and {jr} (joint range) are filled in per run;
# the integrator is set post-compile (see module docstring) so explicit Euler is
# not upgraded away. compiler angle=radian so {jr} is interpreted in radians.
MJCF = """
<mujoco model="momentum_test">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 0"/>
  <default><joint damping="0.2" armature="0.01"/></default>
  <worldbody>
    <body name="bus" pos="0 0 0">
      <freejoint name="base"/>
      <geom type="box" size="0.4 0.4 0.4" mass="40"/>
      <body name="link1" pos="0.4 0 0">
        <joint name="shoulder" type="hinge" axis="0 1 0"{jr}/>
        <geom type="capsule" fromto="0 0 0 1.0 0 0" size="0.05" mass="8"/>
        <body name="link2" pos="1.0 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0"{jr}/>
          <geom type="capsule" fromto="0 0 0 1.0 0 0" size="0.04" mass="8"/>
          <body name="ee" pos="1.0 0 0"><geom type="sphere" size="0.09" mass="4"/></body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder" joint="shoulder" kp="30" kv="15"/>
    <position name="elbow"    joint="elbow"    kp="30" kv="10"/>
  </actuator>
  <mjorbit use_j2="false" use_drag="false" use_srp="false" use_magnetic="false"
           use_gravity_gradient="false"/>
</mujoco>
"""

_INTEGRATORS = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
}

R0 = R_EARTH + 400.0
V0 = float(np.sqrt(GM_EARTH / R0))
ARM_POSE = [0.6, -1.0]                 # held bent so the body is asymmetric (rad)
SPIN_BODY = [0.4, 0.6, 0.3]            # initial bus angular velocity (body frame, rad/s)
LIMIT_RAD = 0.5                        # regime (B) joint range; ARM_POSE violates it


def build(integrator: str, dt: float, joint_limit: float | None = None) -> MjoModel:
    """Compile the model, then pin the integrator post-compile (see docstring).

    ``joint_limit`` None -> no joint range (regime A); a float L -> ``range``
    +-L rad on both hinges (regime B). With ARM_POSE outside +-L the limit
    constraint is active and penetrating.
    """
    jr = f' range="{-joint_limit} {joint_limit}"' if joint_limit is not None else ""
    text = MJCF.format(dt=dt, jr=jr)
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(text)
        path = Path(f.name)
    try:
        model = MjoModel.from_xml_path(str(path))
    finally:
        path.unlink(missing_ok=True)
    model.opt.integrator = int(_INTEGRATORS[integrator])
    return model


def total_angmom(model: MjoModel, data) -> np.ndarray:
    """System angular momentum about the system CoM, world (ECI-parallel) axes."""
    mass = np.asarray(model.body_mass)[1:]
    c = np.asarray(data.xipos)[1:]
    cv = np.asarray(data.cvel)[1:]
    w = cv[:, 0:3]                       # body angular velocity (world axes)
    vref = cv[:, 3:6]                    # linear vel referenced at the subtree CoM
    Idiag = np.asarray(model.body_inertia)[1:]
    R = np.asarray(data.ximat)[1:].reshape(-1, 3, 3)
    ccm = (mass[:, None] * c).sum(0) / mass.sum()
    L = np.zeros(3)
    for i in range(len(mass)):
        v_com = vref[i] + np.cross(w[i], c[i] - ccm)   # transport to body CoM
        Iw = R[i] @ np.diag(Idiag[i]) @ R[i].T
        L += Iw @ w[i] + mass[i] * np.cross(c[i] - ccm, v_com)
    return L


def simulate(integrator: str, dt: float, joint_limit: float | None = None,
             arm_pose=ARM_POSE, duration: float = 30.0):
    """Free tumble from a known L0; return (times, relative drift |L-L0|/|L0|)."""
    model = build(integrator, dt, joint_limit)
    data = model.make_data(orbit=OrbitInit(R_eci=[R0, 0, 0], V_eci=[0, V0, 0]))
    data.qpos[7:9] = arm_pose
    np.copyto(data.ctrl, data.qpos[7:9])     # hold the arm (quasi-rigid tumble)
    data.qvel[3:6] = SPIN_BODY               # spin the bus up
    mjo_forward(model, data)
    L0 = total_angmom(model, data)
    n0 = np.linalg.norm(L0)
    n = int(round(duration / dt))
    rec = max(1, int(round(0.5 / dt)))
    ts, drift = [0.0], [0.0]
    for k in range(n):
        mjo_step(model, data)
        if k % rec == 0 or k == 0:
            # cvel/xipos lag qpos/qvel by one step right after mjo_step (computed
            # at the start of mj_step), so refresh forward kinematics before reading
            # them. forward() does not integrate, so the trajectory is unchanged.
            mjo_forward(model, data)
            ts.append(float(data.time))
            drift.append(float(np.linalg.norm(total_angmom(model, data) - L0) / n0))
    return np.array(ts), np.array(drift), n0


def main() -> None:
    integrators = ["Euler", "implicit", "implicitfast", "RK4"]
    dt = 0.01
    print("=" * 76)
    print("Angular-momentum conservation vs integrator (free tumble, external torques OFF)")
    print("conservation => |L(t)-L0|/|L0| stays 0")
    print("=" * 76)

    free, limited = {}, {}
    print(f"\n(A) joints FREE (no limits):           "
          f"{'1st-step':>10}{'max@30s':>10}")
    for ig in integrators:
        t, d, n0 = simulate(ig, dt, joint_limit=None)
        free[ig] = (t, d)
        print(f"    {ig:13s}                      {d[1]*100:9.3f}%{d.max()*100:9.3f}%")

    print(f"\n(B) joint limit +-{LIMIT_RAD} rad VIOLATED by the {ARM_POSE} rad pose "
          f"(constraint active):")
    print(f"    {'integrator':17s}              {'1st-step':>10}{'max@30s':>10}")
    for ig in integrators:
        t, d, n0 = simulate(ig, dt, joint_limit=LIMIT_RAD)
        limited[ig] = (t, d)
        print(f"    {ig:13s}                      {d[1]*100:9.3f}%{d.max()*100:9.3f}%")

    print("\n(C) Euler drift explodes with joint-limit penetration depth (limit +-0.5 rad):")
    for pose in ([0.55, -0.55], [0.7, -0.7], [1.0, -1.0], [1.5, -1.5]):
        _, d, _ = simulate("Euler", dt, joint_limit=0.5, arm_pose=pose)
        print(f"    pose {str(pose):14s} (penetration {pose[0]-0.5:.2f} rad): "
              f"max = {d.max()*100:8.2f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("\n(matplotlib not available; run with `pixi run -e report` to save the plot)")
        return

    colors = {"Euler": "tab:red", "implicit": "tab:orange",
              "implicitfast": "tab:green", "RK4": "tab:blue"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle("Angular-momentum conservation by integrator (free tumble, no external torque)")
    for ig in integrators:
        t, d = free[ig]
        ax[0].plot(t, d * 100, color=colors[ig], label=ig)
    ax[0].set(title=f"(A) joints free -- benign, dt={dt}",
              xlabel="time [s]", ylabel="drift |L-L0|/|L0| [%]")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    for ig in integrators:
        t, d = limited[ig]
        ax[1].plot(t, d * 100, color=colors[ig], label=ig)
    ax[1].set(title=f"(B) violated joint limit -- first-order integrators blow up, dt={dt}",
              xlabel="time [s]", ylabel="drift |L-L0|/|L0| [%]")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).parent / "momentum_conservation_integrators.png"
    fig.savefig(out, dpi=110)
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
