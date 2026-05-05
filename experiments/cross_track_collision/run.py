"""Offset cross-track collision experiment using the mujoco_orbit simulator API.

Two equal boxes start with a cross-track closing velocity and a small lateral
offset.  The lateral offset is still inside the box contact footprint, so the
boxes initially collide.  The relative state also includes a bounded in-plane CW
component, so on the next cross-track return the boxes have lateral clearance
and pass around each other rather than colliding again.

The simulation is run continuously with ``mjo_step`` from before impact through
one chief orbit.  For interpretation, the plots include two references seeded
from the simulator state just after the first contact ends:

- a zero-g constant-velocity reference, where the boxes drift apart forever
- the full CW relative-motion solution

Figures are saved under ``experiments/cross_track_collision/figures``.

Usage:
    pixi run python experiments/cross_track_collision/run.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mujoco_orbit import MjoData, MjoModel, MjoSpec, OrbitInit, mjo_forward, mjo_step
from tests.mujoco_orbit.reference.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.testdata import TWO_BODIES_XML

EXPERIMENT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXPERIMENT_DIR / "figures"

ALT_KM = 400.0
INCLINATION_DEG = 51.6
TIMESTEP = 0.02
INITIAL_RELATIVE_POSITION_M = np.array([0.45, 0.45, 4.0])
INITIAL_RELATIVE_RADIAL_VELOCITY_M_S = -1.5e-4
IMPACT_SPEED_M_S = 5.0
BOX_HALF_SIZE_M = 0.3
CONTACT_FOOTPRINT_M = 2.0 * BOX_HALF_SIZE_M
SIM_ORBIT_FRACTION = 1.0


@dataclass(frozen=True)
class Trajectory:
    """Recorded relative trajectory from one continuous simulator run."""

    time: np.ndarray
    rel_pos: np.ndarray
    rel_vel: np.ndarray
    contact_count: np.ndarray
    mean_motion: float


def cw_relative(
    pos0: np.ndarray, vel0: np.ndarray, mean_motion: float, dt: np.ndarray
) -> np.ndarray:
    """Clohessy-Wiltshire relative position for elapsed time ``dt``."""
    phase = mean_motion * dt
    cn = np.cos(phase)
    sn = np.sin(phase)

    x0, y0, z0 = pos0
    vx0, vy0, vz0 = vel0
    n = mean_motion

    x = (4.0 - 3.0 * cn) * x0 + sn / n * vx0 + 2.0 / n * (1.0 - cn) * vy0
    y = (
        6.0 * (sn - phase) * x0
        + y0
        - 2.0 / n * (1.0 - cn) * vx0
        + (4.0 * sn - 3.0 * phase) / n * vy0
    )
    z = z0 * cn + vz0 / n * sn
    return np.column_stack((x, y, z))


def relative_position(data: MjoData) -> np.ndarray:
    """Return body_b - body_a position in the chief-centered LVLH frame."""
    pos_a = data.lvlh_position_from_world(data.qpos[0:3])
    pos_b = data.lvlh_position_from_world(data.qpos[7:10])
    return pos_b - pos_a


def relative_velocity(data: MjoData) -> np.ndarray:
    """Return body_b - body_a translational velocity in the LVLH frame."""
    vel_a = data.lvlh_velocity_from_world(data.qpos[0:3], data.qvel[0:3])
    vel_b = data.lvlh_velocity_from_world(data.qpos[7:10], data.qvel[6:9])
    return vel_b - vel_a


def configure_cross_track_collision(data: MjoData, mean_motion: float) -> None:
    """Place the boxes on an offset cross-track collision course."""
    rel_pos = INITIAL_RELATIVE_POSITION_M
    rel_vel = np.array(
        [
            INITIAL_RELATIVE_RADIAL_VELOCITY_M_S,
            -2.0 * mean_motion * rel_pos[0],
            -2.0 * IMPACT_SPEED_M_S,
        ]
    )
    pos_a_lvlh = -0.5 * rel_pos
    pos_b_lvlh = 0.5 * rel_pos
    vel_a_lvlh = -0.5 * rel_vel
    vel_b_lvlh = 0.5 * rel_vel

    data.qpos[0:3] = data.world_position_from_lvlh(pos_a_lvlh)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:10] = data.world_position_from_lvlh(pos_b_lvlh)
    data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]

    data.qvel[:] = 0.0
    data.qvel[0:3] = data.world_velocity_from_lvlh(pos_a_lvlh, vel_a_lvlh)
    data.qvel[6:9] = data.world_velocity_from_lvlh(pos_b_lvlh, vel_b_lvlh)


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
    spec = MjoSpec.from_xml_path(TWO_BODIES_XML)
    spec.mjorbit.use_j2 = False
    spec.mjorbit.use_drag = False
    spec.mjorbit.use_srp = False
    spec.mjorbit.use_magnetic = False
    return spec.compile(mj_timestep=TIMESTEP)


def run_orbit_simulation() -> Trajectory:
    """Run one continuous orbit-coupled collision trajectory with ``mjo_step``."""
    orbit_init, mean_motion = make_orbit_init()
    model = make_model()
    data = MjoData(model, orbit=orbit_init)
    configure_cross_track_collision(data, mean_motion)
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


def contact_intervals(contact_count: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index intervals where contact is active."""
    contact_indices = np.flatnonzero(contact_count > 0)
    if contact_indices.size == 0:
        return []

    intervals: list[tuple[int, int]] = []
    start = contact_indices[0]
    previous = contact_indices[0]
    for idx in contact_indices[1:]:
        if idx == previous + 1:
            previous = idx
        else:
            intervals.append((start, previous))
            start = idx
            previous = idx
    intervals.append((start, previous))
    return intervals


def first_post_contact_index(contact_count: np.ndarray) -> int:
    """Return the first sample after the initial contact interval ends."""
    intervals = contact_intervals(contact_count)
    if not intervals:
        raise RuntimeError("No contact was detected in the orbit-coupled simulation.")

    post_contact_idx = intervals[0][1] + 1
    if post_contact_idx >= contact_count.size:
        raise RuntimeError("Contact did not end before the simulation finished.")
    return post_contact_idx


def post_contact_references(trajectory: Trajectory, post_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-g and CW references seeded from the same post-impact state."""
    zero_g = np.full_like(trajectory.rel_pos, np.nan, dtype=float)
    cw = np.full_like(trajectory.rel_pos, np.nan, dtype=float)

    t0 = trajectory.time[post_idx]
    pos0 = trajectory.rel_pos[post_idx]
    vel0 = trajectory.rel_vel[post_idx]
    dt = trajectory.time[post_idx:] - t0

    zero_g[post_idx:] = pos0 + dt[:, None] * vel0
    cw[post_idx:] = cw_relative(pos0, vel0, trajectory.mean_motion, dt)
    return zero_g, cw


def shade_contact_intervals(ax, trajectory: Trajectory, *, label_first: bool = False) -> None:
    """Shade contact intervals on a time-axis plot."""
    for interval_idx, (start, end) in enumerate(contact_intervals(trajectory.contact_count)):
        label = "contact" if label_first and interval_idx == 0 else None
        ax.axvspan(
            trajectory.time[start],
            trajectory.time[min(end + 1, trajectory.time.size - 1)],
            color="0.85",
            alpha=0.7,
            label=label,
        )


def save_cross_track_plot(
    trajectory: Trajectory, post_idx: int, zero_g: np.ndarray, cw: np.ndarray
) -> Path:
    """Save relative cross-track position vs time."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(trajectory.time, trajectory.rel_pos[:, 2], label="mujoco_orbit")
    ax.plot(trajectory.time, zero_g[:, 2], "--", label="zero-g reference")
    ax.plot(trajectory.time, cw[:, 2], ":", linewidth=2.2, label="CW reference")
    ax.axhline(CONTACT_FOOTPRINT_M, color="0.55", linestyle="-.", linewidth=1.0)
    ax.axhline(-CONTACT_FOOTPRINT_M, color="0.55", linestyle="-.", linewidth=1.0)
    shade_contact_intervals(ax, trajectory, label_first=True)
    ax.set_title("Cross-track separation after offset impact")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    path = FIGURE_DIR / "cross_track_separation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_phase_plot(trajectory: Trajectory, post_idx: int) -> Path:
    """Save the cross-track phase plane, where the bounded z motion is circular."""
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


def save_lateral_clearance_plot(trajectory: Trajectory) -> Path:
    """Save lateral offsets relative to the box contact footprint."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(trajectory.time, trajectory.rel_pos[:, 0], label="x radial")
    ax.plot(trajectory.time, trajectory.rel_pos[:, 1], label="y along-track")
    ax.axhline(CONTACT_FOOTPRINT_M, color="0.45", linestyle="--", linewidth=1.0)
    ax.axhline(-CONTACT_FOOTPRINT_M, color="0.45", linestyle="--", linewidth=1.0)
    shade_contact_intervals(ax, trajectory, label_first=True)
    ax.set_title("Lateral clearance at cross-track returns")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative lateral offset [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    path = FIGURE_DIR / "lateral_clearance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_relative_trajectory_plot(trajectory: Trajectory, post_idx: int, cw: np.ndarray) -> Path:
    """Save the 3D relative trajectory after first contact."""
    fig = plt.figure(figsize=(7.0, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        trajectory.rel_pos[post_idx:, 0],
        trajectory.rel_pos[post_idx:, 1],
        trajectory.rel_pos[post_idx:, 2],
        label="mujoco_orbit",
    )
    ax.plot(cw[post_idx:, 0], cw[post_idx:, 1], cw[post_idx:, 2], "--", label="CW reference")
    ax.scatter(
        [trajectory.rel_pos[post_idx, 0]],
        [trajectory.rel_pos[post_idx, 1]],
        [trajectory.rel_pos[post_idx, 2]],
        s=35,
        label="post-impact state",
    )
    ax.set_title("Post-impact relative trajectory")
    ax.set_xlabel("x radial [m]")
    ax.set_ylabel("y along-track [m]")
    ax.set_zlabel("z cross-track [m]")
    ax.legend(loc="best")

    path = FIGURE_DIR / "relative_trajectory_3d.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_component_plot(trajectory: Trajectory, post_idx: int) -> Path:
    """Save all relative position components."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(trajectory.time, trajectory.rel_pos[:, 0], label="x radial")
    ax.plot(trajectory.time, trajectory.rel_pos[:, 1], label="y along-track")
    ax.plot(trajectory.time, trajectory.rel_pos[:, 2], label="z cross-track")
    ax.axvline(trajectory.time[post_idx], color="0.35", linestyle="--", label="post-contact")
    shade_contact_intervals(ax, trajectory)
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
        save_lateral_clearance_plot(trajectory),
        save_relative_trajectory_plot(trajectory, post_idx, cw),
        save_component_plot(trajectory, post_idx),
    ]

    print("Saved figures:")
    for path in paths:
        print(f"  {path.relative_to(EXPERIMENT_DIR)}")


if __name__ == "__main__":
    main()
