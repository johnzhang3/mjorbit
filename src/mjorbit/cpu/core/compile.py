"""compile_cpu — load MuJoCo model and build a CPUScenario from config."""

from __future__ import annotations

import numpy as np
import mujoco

from mjorbit.cpu.core.config import CPUScenarioCfg
from mjorbit.cpu.core.scenario import CPUScenario, SurfaceMetadata, MagneticMetadata
from mjorbit.cpu.core.actuators import ActuatorState
from mjorbit.cpu.orbit.state import OrbitState, FrameCache, EnvironmentCache
from mjorbit.cpu.orbit.lvlh import update_frame_cache
from mjorbit.cpu.orbit.environment import update_environment_cache


def compile_cpu(cfg: CPUScenarioCfg) -> CPUScenario:
    """Load model and build a CPUScenario ready for stepping."""
    mjm = mujoco.MjModel.from_xml_path(cfg.mujoco.xml_path)

    # Override timestep if specified
    if cfg.mujoco.dt is not None:
        mjm.opt.timestep = cfg.mujoco.dt

    # Disable built-in gravity — orbital environment provides all gravity
    mjm.opt.gravity[:] = 0.0

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    # Build orbit state
    orbit = OrbitState(
        R_eci=cfg.orbit.R_eci.copy(),
        V_eci=cfg.orbit.V_eci.copy(),
        t=cfg.orbit.t0,
    )

    # Build frame and environment caches
    frame_cache = update_frame_cache(orbit, use_j2=cfg.use_j2)
    env_cache = update_environment_cache(orbit, frame_cache)

    # Resolve surface metadata
    surfaces: list[SurfaceMetadata] = []
    for s in cfg.surfaces:
        body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, s.body_name)
        if body_id < 0:
            raise ValueError(f"Surface body '{s.body_name}' not found in MuJoCo model")
        surfaces.append(
            SurfaceMetadata(
                body_id=body_id,
                center_of_pressure_body=s.center_of_pressure_body.copy(),
                normal_body=s.normal_body / np.linalg.norm(s.normal_body),
                area=s.area,
                drag_coeff=s.drag_coeff,
                srp_coeff=s.srp_coeff,
                use_drag=s.use_drag,
                use_srp=s.use_srp,
            )
        )

    # Resolve magnetic body metadata
    magnetic_bodies: list[MagneticMetadata] = []
    for m in cfg.magnetic_bodies:
        body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, m.body_name)
        if body_id < 0:
            raise ValueError(f"Magnetic body '{m.body_name}' not found in MuJoCo model")
        magnetic_bodies.append(
            MagneticMetadata(body_id=body_id, dipole_body=m.dipole_body.copy())
        )

    # Build actuator state
    n_rw = len(cfg.reaction_wheels)
    rw_inertia = np.array([rw.inertia for rw in cfg.reaction_wheels]) if n_rw > 0 else np.zeros(0)
    n_mtq = len(cfg.magnetorquers)
    n_thr = len(cfg.thrusters)
    actuator_state = ActuatorState.zeros(n_rw, rw_inertia, n_mtq, n_thr)

    return CPUScenario(
        mjm=mjm,
        mjd=mjd,
        orbit=orbit,
        frame_cache=frame_cache,
        env_cache=env_cache,
        surfaces=surfaces,
        magnetic_bodies=magnetic_bodies,
        actuator_state=actuator_state,
        cfg=cfg,
    )
