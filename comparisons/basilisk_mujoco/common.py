"""Shared utilities for Basilisk-MuJoCo comparison scenarios."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_orbit import MjoModel
from mujoco_orbit.constants import GM_EARTH, R_EARTH

HARNESS_DIR = Path(__file__).resolve().parent
ASSET_DIR = HARNESS_DIR / "assets"
OUT_DIR = HARNESS_DIR / "out"


@dataclass(frozen=True)
class CircularOrbit:
    """Circular orbit initial state and derived constants."""

    alt_km: float
    inc_rad: float
    radius_km: float
    mean_motion_rad_s: float
    period_s: float
    r_eci_km: np.ndarray
    v_eci_km_s: np.ndarray


def make_circular_orbit(alt_km: float = 400.0, inc_rad: float | None = None) -> CircularOrbit:
    """Return the canonical circular LEO used by the comparison cases."""
    if inc_rad is None:
        inc_rad = np.deg2rad(51.6)
    radius_km = R_EARTH + alt_km
    mean_motion = float(np.sqrt(GM_EARTH / radius_km**3))
    period_s = float(2.0 * np.pi / mean_motion)
    speed_km_s = float(np.sqrt(GM_EARTH / radius_km))
    r_eci_km = np.array([radius_km, 0.0, 0.0], dtype=np.float64)
    v_eci_km_s = np.array(
        [0.0, speed_km_s * np.cos(inc_rad), speed_km_s * np.sin(inc_rad)],
        dtype=np.float64,
    )
    return CircularOrbit(
        alt_km=alt_km,
        inc_rad=inc_rad,
        radius_km=radius_km,
        mean_motion_rad_s=mean_motion,
        period_s=period_s,
        r_eci_km=r_eci_km,
        v_eci_km_s=v_eci_km_s,
    )


def circular_orbit_state_at(
    times_s: Iterable[float],
    *,
    alt_km: float = 400.0,
    inc_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact two-body circular ECI position and velocity samples."""
    orbit = make_circular_orbit(alt_km=alt_km, inc_rad=inc_rad)
    times = np.asarray(list(times_s), dtype=np.float64)
    phase = orbit.mean_motion_rad_s * times
    c = np.cos(phase)
    s = np.sin(phase)
    speed = orbit.mean_motion_rad_s * orbit.radius_km

    r_eci = np.column_stack(
        (
            orbit.radius_km * c,
            orbit.radius_km * s * np.cos(orbit.inc_rad),
            orbit.radius_km * s * np.sin(orbit.inc_rad),
        )
    )
    v_eci = np.column_stack(
        (
            -speed * s,
            speed * c * np.cos(orbit.inc_rad),
            speed * c * np.sin(orbit.inc_rad),
        )
    )
    return r_eci, v_eci


def sample_steps(n_steps: int, max_samples: int) -> np.ndarray:
    """Return unique step indices including the first and last step."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    return np.unique(np.linspace(0, n_steps, min(n_steps + 1, max_samples), dtype=int))


def compile_mjorbit_model(
    xml_path: Path,
    *,
    plugin_body: str,
    mj_timestep: float,
    orbit_dt: float | None = None,
    use_j2: bool = False,
    use_drag: bool = False,
    use_srp: bool = False,
    use_magnetic: bool = False,
    use_gravity_gradient: bool = False,
) -> MjoModel:
    """Compile an MJCF file with an injected ``<mjorbit>`` block."""
    attrs = [
        f'plugin_body="{plugin_body}"',
        f'use_j2="{_xml_bool(use_j2)}"',
        f'use_drag="{_xml_bool(use_drag)}"',
        f'use_srp="{_xml_bool(use_srp)}"',
        f'use_magnetic="{_xml_bool(use_magnetic)}"',
        f'use_gravity_gradient="{_xml_bool(use_gravity_gradient)}"',
    ]
    if orbit_dt is not None:
        attrs.append(f'orbit_dt="{float(orbit_dt):.17g}"')
    block = "\n  <mjorbit " + " ".join(attrs) + ">\n  </mjorbit>\n"
    text = xml_path.read_text().replace("</mujoco>", block + "</mujoco>")

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as file:
        file.write(text)
        configured_path = Path(file.name)
    try:
        return MjoModel.from_xml_path(str(configured_path), mj_timestep=mj_timestep)
    finally:
        configured_path.unlink(missing_ok=True)


def body_eci_state(
    data: Any,
    *,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a MuJoCo body COM state in absolute ECI coordinates."""
    r_eci_m = data.eci_position_from_world(data.xipos[body_id])
    v_eci_m_s = data.eci_velocity_from_world(data.cvel[body_id, 3:6])
    return np.asarray(r_eci_m, dtype=np.float64) * 1.0e-3, np.asarray(v_eci_m_s) * 1.0e-3


def system_com_world_m(model: Any, data: Any) -> np.ndarray:
    """Return total modeled body COM in the MuJoCo world frame."""
    masses = np.asarray(model.body_mass[1:model.nbody], dtype=np.float64)
    positions = np.asarray(data.xipos[1:model.nbody], dtype=np.float64)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return np.average(positions, weights=masses, axis=0)


def orbital_energy_km2_s2(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> float:
    """Return two-body specific orbital energy."""
    r = float(np.linalg.norm(r_eci_km))
    v = float(np.linalg.norm(v_eci_km_s))
    return 0.5 * v * v - GM_EARTH / r


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def basilisk_mujoco_import_status() -> tuple[bool, str]:
    """Return whether the Basilisk MuJoCo module can be imported."""
    try:
        from Basilisk.simulation import mujoco as _mujoco  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional external install
        return False, f"{type(exc).__name__}: {exc}"
    return True, "Basilisk.simulation.mujoco import succeeded"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _xml_bool(value: bool) -> str:
    return "true" if value else "false"

