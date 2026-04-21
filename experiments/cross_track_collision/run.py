"""Cross-track collision experiment using the mujoco_orbit simulator API.

Two equal boxes start on the LVLH cross-track axis and collide head-on.  The
simulation is run continuously with ``mjo_step`` from before impact through the
post-impact coast.  For interpretation, the plots include two references seeded
from the simulator state just after contact ends:

- a zero-g constant-velocity reference, where the boxes drift apart forever
- the CW cross-track oscillator, z(t) = z0 cos(n t) + zdot0 / n sin(n t)

Figures are saved under ``experiments/cross_track_collision/figures``.

Usage:
    uv run python experiments/cross_track_collision/run.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward, mjo_step
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import TWO_BODIES_XML

EXPERIMENT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXPERIMENT_DIR / "figures"

ALT_KM = 400.0
INCLINATION_DEG = 51.6
TIMESTEP = 0.02
INITIAL_HALF_SEPARATION_M = 2.0
IMPACT_SPEED_M_S = 5.0
SIM_ORBIT_FRACTION = 1


@dataclass(frozen=True)
class Trajectory:
    """Recorded relative trajectory from one continuous simulator run."""

    time: np.ndarray
    rel_pos: np.ndarray
    rel_vel: np.ndarray
    contact_count: np.ndarray
    mean_motion: float


def cw_cross_track(z0: float, vz0: float, mean_motion: float, dt: np.ndarray) -> np.ndarray:
    """CW cross-track relative position for elapsed time ``dt``."""
    phase = mean_motion * dt
    return z0 * np.cos(phase) + vz0 / mean_motion * np.sin(phase)


def relative_position(data: MjoData) -> np.ndarray:
    """Return body_b - body_a position in the MuJoCo/LVLH frame."""
    return data.qpos[7:10].copy() - data.qpos[0:3].copy()


def relative_velocity(data: MjoData) -> np.ndarray:
    """Return body_b - body_a translational velocity in the MuJoCo/LVLH frame."""
    return data.qvel[6:9].copy() - data.qvel[0:3].copy()


def configure_cross_track_collision(data: MjoData) -> None:
    """Place the two free boxes on the cross-track axis with opposing velocities."""
    data.qpos[0:3] = [0.0, 0.0, -INITIAL_HALF_SEPARATION_M]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:10] = [0.0, 0.0, INITIAL_HALF_SEPARATION_M]
    data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]

    data.qvel[:] = 0.0
    data.qvel[2] = IMPACT_SPEED_M_S
    data.qvel[8] = -IMPACT_SPEED_M_S


def make_orbit_init() -> tuple[OrbitInit, float]:
    """Return a circular LEO initial condition and mean motion."""
    a_km = R_EARTH + ALT_KM
    r_eci, v_eci = keplerian_to_cartesian(
        a=a_km,
        e=0.0,
        inc=np.deg2rad(INCLINATION_DEG),
        raan=0.0,
        argp=0.0,
        nu=0.0,
    )
    mean_motion = np.sqrt(GM_EARTH / a_km**3)
    return OrbitInit(R_eci=r_eci, V_eci=v_eci), mean_motion


def make_model() -> MjoModel:
    """Compile the two-box contact fixture with non-gravitational environment off."""
    return MjoModel.from_xml_path(
        TWO_BODIES_XML,
        mj_timestep=TIMESTEP,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )


def run_orbit_simulation() -> Trajectory:
    """Run one continuous orbit-coupled collision trajectory with ``mjo_step``."""
    orbit_init, mean_motion = make_orbit_init()
    model = make_model()
    data = MjoData(model, orbit=orbit_init)
    configure_cross_track_collision(data)
    mjo_forward(model, data)

    orbit_period = 2.0 * np.pi / mean_motion
    n_steps = int(round(SIM_ORBIT_FRACTION * orbit_period / model.opt.timestep))

    time = np.empty(n_steps + 1)
    rel_pos = np.empty((n_steps + 1, 3))
    rel_vel = np.empty((n_steps + 1, 3))
    contact_count = np.empty(n_steps + 1, dtype=int)

    for step_idx in range(n_steps + 1):
        time[step_idx] = step_idx * model.opt.timestep
        rel_pos[step_idx] = relative_position(data)
        rel_vel[step_idx] = relative_velocity(data)
        contact_count[step_idx] = data.ncon

        if step_idx < n_steps:
            mjo_step(model, data)

    return Trajectory(
        time=time,
        rel_pos=rel_pos,
        rel_vel=rel_vel,
        contact_count=contact_count,
        mean_motion=mean_motion,
    )


def first_post_contact_index(contact_count: np.ndarray) -> int:
    """Return the first sample after the initial contact interval ends."""
    contact_indices = np.flatnonzero(contact_count > 0)
    if contact_indices.size == 0:
        raise RuntimeError("No contact was detected in the orbit-coupled simulation.")

    last_initial_contact = contact_indices[0]
    for idx in contact_indices[1:]:
        if idx == last_initial_contact + 1:
            last_initial_contact = idx
            continue
        break

    post_contact_idx = last_initial_contact + 1
    if post_contact_idx >= contact_count.size:
        raise RuntimeError("Contact did not end before the simulation finished.")
    return post_contact_idx


def post_contact_references(trajectory: Trajectory, post_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-g and CW cross-track references seeded from the same post-impact state."""
    zero_g = np.full_like(trajectory.time, np.nan, dtype=float)
    cw = np.full_like(trajectory.time, np.nan, dtype=float)

    t0 = trajectory.time[post_idx]
    z0 = trajectory.rel_pos[post_idx, 2]
    vz0 = trajectory.rel_vel[post_idx, 2]
    dt = trajectory.time[post_idx:] - t0

    zero_g[post_idx:] = z0 + vz0 * dt
    cw[post_idx:] = cw_cross_track(z0, vz0, trajectory.mean_motion, dt)
    return zero_g, cw


def save_cross_track_plot(
    trajectory: Trajectory, post_idx: int, zero_g: np.ndarray, cw: np.ndarray
) -> Path:
    """Save relative cross-track position vs time."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(trajectory.time, trajectory.rel_pos[:, 2], label="mujoco_orbit")
    ax.plot(trajectory.time, zero_g, "--", label="zero-g reference")
    ax.plot(trajectory.time, cw, ":", linewidth=2.2, label="CW reference")
    ax.axvspan(
        trajectory.time[np.argmax(trajectory.contact_count > 0)],
        trajectory.time[post_idx],
        color="0.85",
        alpha=0.7,
        label="contact",
    )
    ax.set_title("Cross-track separation after impact")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    path = FIGURE_DIR / "cross_track_separation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_phase_plot(trajectory: Trajectory, post_idx: int) -> Path:
    """Save the cross-track phase plane, where the bounded motion is circular."""
    z0 = trajectory.rel_pos[post_idx, 2]
    vz0 = trajectory.rel_vel[post_idx, 2]
    phase_radius = np.hypot(z0, vz0 / trajectory.mean_motion)
    theta = np.linspace(0.0, 2.0 * np.pi, 512)

    fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)
    ax.plot(
        trajectory.rel_pos[post_idx:, 2],
        trajectory.rel_vel[post_idx:, 2] / trajectory.mean_motion,
        label="mujoco_orbit",
    )
    ax.plot(
        phase_radius * np.cos(theta),
        phase_radius * np.sin(theta),
        "--",
        label="CW phase radius",
    )
    ax.scatter([z0], [vz0 / trajectory.mean_motion], s=35, label="post-impact state")
    ax.set_title("Cross-track phase plane")
    ax.set_xlabel("relative z [m]")
    ax.set_ylabel("relative zdot / n [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    path = FIGURE_DIR / "cross_track_phase.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_component_plot(trajectory: Trajectory, post_idx: int) -> Path:
    """Save all relative position components to show the collision is cross-track."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(trajectory.time, trajectory.rel_pos[:, 0], label="x radial")
    ax.plot(trajectory.time, trajectory.rel_pos[:, 1], label="y along-track")
    ax.plot(trajectory.time, trajectory.rel_pos[:, 2], label="z cross-track")
    ax.axvline(trajectory.time[post_idx], color="0.35", linestyle="--", label="post-contact")
    ax.set_title("Relative position components")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative position [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    path = FIGURE_DIR / "relative_components.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    trajectory = run_orbit_simulation()
    post_idx = first_post_contact_index(trajectory.contact_count)
    zero_g, cw = post_contact_references(trajectory, post_idx)

    paths = [
        save_cross_track_plot(trajectory, post_idx, zero_g, cw),
        save_phase_plot(trajectory, post_idx),
        save_component_plot(trajectory, post_idx),
    ]

    print("Saved figures:")
    for path in paths:
        print(f"  {path.relative_to(EXPERIMENT_DIR)}")


if __name__ == "__main__":
    main()
