"""Cross-track collision: orbit-coupled vs. naive zero-g MuJoCo.

Runs the same collision scenario as ``run.py`` twice from identical
world-frame initial state:

- ``mujoco_orbit``: ``mjo_step`` with the full Encke differential-gravity
  coupling.
- naive zero-g: plain ``mujoco.mj_step`` on the same XML (gravity already
  ``0 0 0``), with no orbital coupling at all.

Cross-track position is reported as the projection of each body's
MuJoCo world-frame position onto the chief's orbit normal evaluated at
``t=0``. For a circular Keplerian chief, this projection is constant in
ECI and equal to the LVLH cross-track component, which is verified
numerically against ``MjoData.lvlh_position_from_world``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mujoco_orbit import MjoData, MjoModel, mjo_forward, mjo_step
from mujoco_orbit.testdata import TWO_BODIES_XML

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (  # noqa: E402
    SIM_ORBIT_FRACTION,
    TIMESTEP,
    Trajectory,
    configure_cross_track_collision,
    contact_intervals,
    make_model,
    make_orbit_init,
    relative_position,
    relative_velocity,
)


@dataclass(frozen=True)
class ComparisonResult:
    """Synchronized cross-track samples for both rollouts."""

    time: np.ndarray
    cross_track_a_orbit: np.ndarray
    cross_track_b_orbit: np.ndarray
    cross_track_a_naive: np.ndarray
    cross_track_b_naive: np.ndarray
    contact_count_orbit: np.ndarray
    mean_motion: float
    orbit_period: float
    h_hat_world: np.ndarray
    rel_post_contact: tuple[np.ndarray, np.ndarray] | None


def _orbit_normal(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    h = np.cross(r_eci, v_eci)
    return h / np.linalg.norm(h)


def _run_orbit(
    n_steps: int, h_hat: np.ndarray
) -> tuple[Trajectory, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orbit_init, mean_motion = make_orbit_init()
    model = make_model()
    data = MjoData(model, orbit=orbit_init)
    configure_cross_track_collision(data, mean_motion)
    mjo_forward(model, data)

    qpos0 = np.array(data.qpos, dtype=float, copy=True)
    qvel0 = np.array(data.qvel, dtype=float, copy=True)

    time = np.empty(n_steps + 1)
    rel_pos = np.empty((n_steps + 1, 3))
    rel_vel = np.empty((n_steps + 1, 3))
    contact_count = np.empty(n_steps + 1, dtype=int)
    z_a = np.empty(n_steps + 1)
    z_b = np.empty(n_steps + 1)
    z_a_lvlh = np.empty(n_steps + 1)
    z_b_lvlh = np.empty(n_steps + 1)

    for step_idx in range(n_steps + 1):
        time[step_idx] = step_idx * model.opt.timestep
        rel_pos[step_idx] = relative_position(data)
        rel_vel[step_idx] = relative_velocity(data)
        contact_count[step_idx] = data.ncon

        pos_a = np.asarray(data.qpos[0:3], dtype=float)
        pos_b = np.asarray(data.qpos[7:10], dtype=float)
        z_a[step_idx] = float(h_hat @ pos_a)
        z_b[step_idx] = float(h_hat @ pos_b)
        z_a_lvlh[step_idx] = float(data.lvlh_position_from_world(pos_a)[2])
        z_b_lvlh[step_idx] = float(data.lvlh_position_from_world(pos_b)[2])

        if step_idx < n_steps:
            mjo_step(model, data)

    trajectory = Trajectory(
        time=time,
        rel_pos=rel_pos,
        rel_vel=rel_vel,
        contact_count=contact_count,
        mean_motion=mean_motion,
    )
    return trajectory, qpos0, qvel0, np.column_stack((z_a, z_b)), np.column_stack(
        (z_a_lvlh, z_b_lvlh)
    )


def _run_naive(
    qpos0: np.ndarray, qvel0: np.ndarray, n_steps: int, h_hat: np.ndarray
) -> np.ndarray:
    mj_model = mujoco.MjModel.from_xml_path(str(TWO_BODIES_XML))
    mj_model.opt.timestep = TIMESTEP
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos0
    mj_data.qvel[:] = qvel0
    mujoco.mj_forward(mj_model, mj_data)

    z_a = np.empty(n_steps + 1)
    z_b = np.empty(n_steps + 1)
    for step_idx in range(n_steps + 1):
        pos_a = np.asarray(mj_data.qpos[0:3], dtype=float)
        pos_b = np.asarray(mj_data.qpos[7:10], dtype=float)
        z_a[step_idx] = float(h_hat @ pos_a)
        z_b[step_idx] = float(h_hat @ pos_b)
        if step_idx < n_steps:
            mujoco.mj_step(mj_model, mj_data)
    return np.column_stack((z_a, z_b))


def run_both() -> ComparisonResult:
    """Run both rollouts and return the synchronized cross-track samples."""
    orbit_init, mean_motion = make_orbit_init()
    h_hat = _orbit_normal(np.asarray(orbit_init.R_eci), np.asarray(orbit_init.V_eci))
    orbit_period = 2.0 * np.pi / mean_motion
    n_steps = int(round(SIM_ORBIT_FRACTION * orbit_period / TIMESTEP))

    trajectory, qpos0, qvel0, z_orbit, z_orbit_lvlh = _run_orbit(n_steps, h_hat)

    err_lvlh = np.max(np.abs(z_orbit - z_orbit_lvlh))
    if err_lvlh > 1e-6:
        raise AssertionError(
            f"orbit-normal projection disagrees with LVLH-z by {err_lvlh:.3e} m"
        )

    z_naive = _run_naive(qpos0, qvel0, n_steps, h_hat)

    intervals = contact_intervals(trajectory.contact_count)
    if not intervals:
        raise RuntimeError("no contact detected in orbit-coupled rollout")
    post_idx = intervals[0][1] + 1

    rel_post_pos = trajectory.rel_pos[post_idx].copy()
    rel_post_vel = trajectory.rel_vel[post_idx].copy()

    orbit_max_abs = float(np.max(np.abs(trajectory.rel_pos[post_idx:, 2])))
    naive_rel_z = z_naive[:, 1] - z_naive[:, 0]
    naive_end_abs = float(abs(naive_rel_z[-1]))
    if not (naive_end_abs > 2.0 * orbit_max_abs):
        raise AssertionError(
            f"naive zero-g separation at t=T={trajectory.time[-1]:.0f}s should exceed "
            f"the orbit-coupled cross-track amplitude by >2x, but got "
            f"naive_end={naive_end_abs:.3f} m vs orbit_max={orbit_max_abs:.3f} m"
        )

    return ComparisonResult(
        time=trajectory.time,
        cross_track_a_orbit=z_orbit[:, 0],
        cross_track_b_orbit=z_orbit[:, 1],
        cross_track_a_naive=z_naive[:, 0],
        cross_track_b_naive=z_naive[:, 1],
        contact_count_orbit=trajectory.contact_count,
        mean_motion=mean_motion,
        orbit_period=orbit_period,
        h_hat_world=h_hat,
        rel_post_contact=(rel_post_pos, rel_post_vel),
    )


def main() -> None:
    """Print summary statistics for a quick command-line sanity check."""
    result = run_both()
    print("Cross-track zero-g vs. orbit-coupled comparison")
    print(f"  orbit period        : {result.orbit_period:.2f} s")
    print(f"  samples per series  : {result.time.size}")
    print(f"  h_hat (world)       : {result.h_hat_world}")
    print(
        "  mujoco_orbit z range: "
        f"A=[{result.cross_track_a_orbit.min():.3f}, {result.cross_track_a_orbit.max():.3f}]  "
        f"B=[{result.cross_track_b_orbit.min():.3f}, {result.cross_track_b_orbit.max():.3f}]"
    )
    print(
        "  naive zero-g z range: "
        f"A=[{result.cross_track_a_naive.min():.3f}, {result.cross_track_a_naive.max():.3f}]  "
        f"B=[{result.cross_track_b_naive.min():.3f}, {result.cross_track_b_naive.max():.3f}]"
    )


if __name__ == "__main__":
    main()
