"""Epoch-anchored time on the warp backend (issue #23).

With ``OrbitInit.epoch`` anchoring, ``t`` is ~1e9 seconds since J2000 — far past
float32 resolution (~64 s at that magnitude). The device therefore splits the
clock into a float64 per-world anchor (``orbit_t0``) plus a float32 relative
clock, and evaluates the time-keyed environment models (sun vector, co-rotating
dipole) in float64. These tests pin CPU/warp parity at epoch scale and the
absolute-time round trip through stepping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

import mjorbit as mjo_cpu
import mjorbit_warp as mjo_warp
from mjorbit.constants import R_EARTH
from mjorbit.testdata import FREE_BODY_XML

# ~2026-04: an epoch-scale t where float32 seconds are ~64 s apart.
T_EPOCH = 8.3e8

_MJORBIT_BLOCK = """
  <mjorbit plugin_body="spacecraft" use_j2="false" use_drag="false" use_srp="false">
    <central_body magnetic_axis="0.3 0 -1"/>
  </mjorbit>
"""


def _write_xml(tmp_path: Path) -> str:
    text = Path(FREE_BODY_XML).read_text()
    path = tmp_path / "model.xml"
    path.write_text(text.replace("</mujoco>", _MJORBIT_BLOCK + "</mujoco>"))
    return str(path)


def _orbit(t: float) -> dict:
    return dict(
        R_eci=np.array([R_EARTH + 550.0, 300.0, -800.0]),
        V_eci=np.array([0.1, 7.4, 0.4]),
        t=t,
    )


def test_env_cache_parity_at_epoch_scale_t(tmp_path: Path):
    xml = _write_xml(tmp_path)

    cpu_model = mjo_cpu.MjoModel.from_xml_path(xml, mj_timestep=0.01)
    cpu_data = cpu_model.make_data(orbit=mjo_cpu.OrbitInit(**_orbit(T_EPOCH)))

    warp_model = mjo_warp.MjoModel.from_xml_path(xml, mj_timestep=0.01)
    warp_data = warp_model.make_data(orbit=mjo_warp.OrbitInit(**_orbit(T_EPOCH)))
    mjo_warp.mjo_forward(warp_model, warp_data)
    mjo_warp.mjo_pull(warp_model, warp_data)

    # float32 device storage of unit-scale vectors: ~1e-7 relative. Anything
    # beyond ~1e-5 means the device evaluated the angles in float32 time.
    np.testing.assert_allclose(
        warp_data.env.sun_vector_eci, cpu_data.env.sun_vector_eci, atol=2e-6
    )
    np.testing.assert_allclose(
        warp_data.env.mag_field_eci,
        cpu_data.env.mag_field_eci,
        rtol=2e-5,
        atol=2e-5 * float(np.linalg.norm(cpu_data.env.mag_field_eci)),
    )


def test_tilted_dipole_corotates_on_device(tmp_path: Path):
    xml = _write_xml(tmp_path)
    warp_model = mjo_warp.MjoModel.from_xml_path(xml, mj_timestep=0.01)

    fields = {}
    for t in (0.0, 3.0e4):
        data = warp_model.make_data(orbit=mjo_warp.OrbitInit(**_orbit(t)))
        mjo_warp.mjo_forward(warp_model, data)
        mjo_warp.mjo_pull(warp_model, data)
        fields[t] = np.asarray(data.env.mag_field_eci, dtype=float).copy()

    delta = np.linalg.norm(fields[0.0] - fields[3.0e4])
    assert delta > 1e-3 * np.linalg.norm(fields[0.0])


def test_absolute_time_round_trips_through_stepping(tmp_path: Path):
    xml = _write_xml(tmp_path)
    warp_model = mjo_warp.MjoModel.from_xml_path(xml, mj_timestep=0.01)
    data = warp_model.make_data(orbit=mjo_warp.OrbitInit(**_orbit(T_EPOCH)))

    n_steps = 100
    for _ in range(n_steps):
        mjo_warp.mjo_step(warp_model, data)
    mjo_warp.mjo_pull(warp_model, data)

    # Absolute time = float64 anchor + small float32 relative clock: the epoch
    # magnitude must not swallow the accumulated 1.0 s (float32 at 8.3e8 would).
    expected = T_EPOCH + n_steps * 0.01
    assert data.orbit.t == pytest.approx(expected, abs=1e-4)


def test_orbit_upload_reanchors_device_clock(tmp_path: Path):
    xml = _write_xml(tmp_path)
    warp_model = mjo_warp.MjoModel.from_xml_path(xml, mj_timestep=0.01)
    data = warp_model.make_data(orbit=mjo_warp.OrbitInit(**_orbit(0.0)))
    for _ in range(10):
        mjo_warp.mjo_step(warp_model, data)

    # Host edit of the device-integrated orbit time requires an explicit upload.
    data.orbit.t = T_EPOCH
    mjo_warp.mjo_upload(warp_model, data, fields=("orbit",))
    for _ in range(10):
        mjo_warp.mjo_step(warp_model, data)
    mjo_warp.mjo_pull(warp_model, data)

    assert data.orbit.t == pytest.approx(T_EPOCH + 0.1, abs=1e-4)
