"""Tests for standard inertial-frame conversion of orbit initial conditions.

The default (``frame="ECI"``, ``epoch=None``) path must be bit-identical to the
pre-feature behavior and must not require astropy. The TEME / epoch paths are guarded
with ``importorskip`` so the suite still passes without the optional ``frames`` extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mjorbit import MjoData, MjoModel, OrbitInit, frames
from mjorbit.data import _resolve_canonical_orbit
from mjorbit.testdata import FREE_BODY_XML

R0 = np.array([7000.0, 1000.0, -500.0])
V0 = np.array([0.5, 7.4, 0.1])
# A concrete, non-J2000 instant so the TEME->GCRF rotation is unambiguously non-trivial.
EPOCH = "2024-03-01T12:00:00"


# ---------------------------------------------------------------------------
# Default path: verbatim, astropy-free
# ---------------------------------------------------------------------------


def test_default_path_is_verbatim_and_does_not_import_astropy(monkeypatch):
    # Poison astropy so any import would raise; the default path must not touch it.
    monkeypatch.setitem(sys.modules, "astropy", None)

    orbit = OrbitInit(R_eci=R0, V_eci=V0, t=123.0)
    R, V, t = frames.resolve_orbit_state(orbit)
    np.testing.assert_array_equal(R, np.asarray(R0, dtype=float))
    np.testing.assert_array_equal(V, np.asarray(V0, dtype=float))
    assert t == 123.0

    # The MjoData-boundary helper takes an even shorter fast path.
    R2, V2, t2 = _resolve_canonical_orbit(orbit)
    np.testing.assert_array_equal(R2, np.asarray(R0, dtype=float))
    np.testing.assert_array_equal(V2, np.asarray(V0, dtype=float))
    assert t2 == 123.0


def test_canonical_aliases_are_identity_rotations():
    for frame in ("ECI", "eci", "GCRF", "GCRS", "ICRF", "J2000", "EME2000"):
        orbit = OrbitInit(R_eci=R0, V_eci=V0, t=7.0, frame=frame)
        R, V, t = frames.resolve_orbit_state(orbit)
        np.testing.assert_array_equal(R, np.asarray(R0, dtype=float))
        np.testing.assert_array_equal(V, np.asarray(V0, dtype=float))
        assert t == 7.0


# ---------------------------------------------------------------------------
# Validation / error paths (astropy-free)
# ---------------------------------------------------------------------------


def test_unknown_frame_raises():
    orbit = OrbitInit(R_eci=R0, V_eci=V0, frame="BANANA")
    with pytest.raises(ValueError, match="Unsupported orbit frame"):
        frames.resolve_orbit_state(orbit)


def test_epoch_dependent_frame_requires_epoch():
    orbit = OrbitInit(R_eci=R0, V_eci=V0, frame="TEME")  # no epoch
    with pytest.raises(ValueError, match="epoch-dependent"):
        frames.resolve_orbit_state(orbit)


# ---------------------------------------------------------------------------
# Epoch -> canonical t (requires astropy)
# ---------------------------------------------------------------------------


def test_seconds_since_j2000_reference_values():
    pytest.importorskip("astropy")
    from astropy.time import Time

    # J2000.0 itself is the origin (2000-01-01 12:00:00 TT, JD 2451545.0).
    assert abs(frames.seconds_since_j2000(Time(2451545.0, format="jd", scale="tt"))) < 1e-6
    # Noon UTC on 2000-01-01 sits TT-UTC = 64.184 s after J2000.0 (TAI-UTC = 32 s in 2000).
    assert abs(frames.seconds_since_j2000("2000-01-01T12:00:00") - 64.184) < 0.5


def test_epoch_anchors_t_without_rotating_for_canonical_frame():
    pytest.importorskip("astropy")
    orbit = OrbitInit(R_eci=R0, V_eci=V0, t=999.0, frame="ECI", epoch=EPOCH)
    R, V, t = frames.resolve_orbit_state(orbit)
    # Canonical frame => no rotation, but t is overridden by the epoch's J2000 offset.
    np.testing.assert_array_equal(R, np.asarray(R0, dtype=float))
    np.testing.assert_array_equal(V, np.asarray(V0, dtype=float))
    assert t == pytest.approx(frames.seconds_since_j2000(EPOCH))
    assert t != 999.0


# ---------------------------------------------------------------------------
# TEME -> GCRF rotation (requires astropy)
# ---------------------------------------------------------------------------


def test_teme_to_gcrf_is_a_nontrivial_norm_preserving_rotation():
    pytest.importorskip("astropy")
    orbit = OrbitInit(R_eci=R0, V_eci=V0, t=0.0, frame="TEME", epoch=EPOCH)
    R, V, t = frames.resolve_orbit_state(orbit)

    # The rotation preserves |R| exactly; |V| is preserved up to the (physical) net
    # precession/nutation frame-rate term astropy correctly includes (~0.05 mm/s),
    # i.e. NOT the spurious 0.46 km/s Earth-rotation term that an ITRS input would risk.
    assert np.linalg.norm(R) == pytest.approx(np.linalg.norm(R0), rel=1e-9)
    assert np.linalg.norm(V) == pytest.approx(np.linalg.norm(V0), rel=1e-6)

    # ... and it actually rotates: TEME-of-date vs GCRF differ by precession since J2000
    # (~0.3 deg over ~24 yr) plus nutation/bias, well above numerical noise.
    cos_angle = np.dot(R, R0) / (np.linalg.norm(R) * np.linalg.norm(R0))
    assert cos_angle < 0.9999999

    # t is anchored to the epoch.
    assert t == pytest.approx(frames.seconds_since_j2000(EPOCH))


def test_teme_round_trip_through_astropy_recovers_input():
    pytest.importorskip("astropy")
    from astropy import units as u
    from astropy.coordinates import GCRS, TEME, CartesianDifferential, CartesianRepresentation
    from astropy.time import Time

    epoch_time = Time(EPOCH)
    orbit = OrbitInit(R_eci=R0, V_eci=V0, frame="TEME", epoch=epoch_time)
    R, V, _ = frames.resolve_orbit_state(orbit)

    # Take the GCRF result back to TEME and confirm we recover the original state.
    rep = CartesianRepresentation(
        R * u.km,
        differentials=CartesianDifferential(V * (u.km / u.s)),
    )
    back = (
        GCRS(rep, obstime=epoch_time)
        .transform_to(TEME(obstime=epoch_time))
        .cartesian
    )
    R_back = back.xyz.to_value(u.km)
    V_back = back.differentials["s"].d_xyz.to_value(u.km / u.s)
    np.testing.assert_allclose(R_back, R0, atol=1e-6)
    np.testing.assert_allclose(V_back, V0, atol=1e-9)


# ---------------------------------------------------------------------------
# Integration: the conversion is applied at the MjoData boundary
# ---------------------------------------------------------------------------


def _model(tmp_path: Path) -> MjoModel:
    text = Path(FREE_BODY_XML).read_text()
    block = (
        '\n  <mjorbit plugin_body="spacecraft" use_j2="false" use_drag="false" '
        'use_srp="false" use_magnetic="false"></mjorbit>\n'
    )
    path = tmp_path / "model.xml"
    path.write_text(text.replace("</mujoco>", block + "</mujoco>"))
    return MjoModel.from_xml_path(str(path), mj_timestep=0.01)


def test_mjodata_default_orbit_is_unchanged(tmp_path: Path):
    model = _model(tmp_path)
    data = MjoData(model, orbit=OrbitInit(R_eci=R0, V_eci=V0, t=42.0))
    np.testing.assert_array_equal(data.orbit.R_eci, np.asarray(R0, dtype=float))
    np.testing.assert_array_equal(data.orbit.V_eci, np.asarray(V0, dtype=float))
    assert data.orbit.t == 42.0


def test_mjodata_applies_teme_conversion_at_boundary(tmp_path: Path):
    pytest.importorskip("astropy")
    model = _model(tmp_path)
    orbit = OrbitInit(R_eci=R0, V_eci=V0, frame="TEME", epoch=EPOCH)
    R_expected, V_expected, t_expected = frames.resolve_orbit_state(orbit)

    data = MjoData(model, orbit=orbit)
    np.testing.assert_allclose(data.orbit.R_eci, R_expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(data.orbit.V_eci, V_expected, rtol=0, atol=1e-12)
    assert data.orbit.t == pytest.approx(t_expected)

    # reset() with no argument must replay the resolved canonical state, not re-rotate.
    data.reset()
    np.testing.assert_allclose(data.orbit.R_eci, R_expected, rtol=0, atol=1e-9)
    assert data.orbit.t == pytest.approx(t_expected)


# ---------------------------------------------------------------------------
# Warp backend re-exports OrbitInit and rebuilds copies; frame/epoch must survive
# so the host CPU MjoData resolves them before the device upload (PR #15 review).
# This needs neither mujoco_warp (the copy is pure Python) nor astropy.
# ---------------------------------------------------------------------------


def test_warp_normalize_orbit_inits_preserves_frame_and_epoch():
    from mjorbit_warp.data import _normalize_orbit_inits

    orbit = OrbitInit(R_eci=R0, V_eci=V0, t=5.0, frame="TEME", epoch=EPOCH)

    single = _normalize_orbit_inits(orbit, nworld=2)
    assert len(single) == 2
    for o in single:
        assert o.frame == "TEME"
        assert o.epoch == EPOCH
        # arrays are still copied, not aliased to the caller's input
        assert not np.shares_memory(o.R_eci, orbit.R_eci)
        assert not np.shares_memory(o.V_eci, orbit.V_eci)

    seq = _normalize_orbit_inits([orbit, orbit], nworld=2)
    assert [o.frame for o in seq] == ["TEME", "TEME"]
    assert [o.epoch for o in seq] == [EPOCH, EPOCH]
