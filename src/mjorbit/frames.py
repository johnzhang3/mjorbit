# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
# pyright: reportMissingImports=false
# (astropy is an optional dependency absent from the default typecheck env, and its
# runtime attributes — Representation.xyz, Differential.d_xyz, Time arithmetic — are
# not described by its type stubs; the behavior is exercised by
# tests/mjorbit/test_frames.py under the `frames` environment.)

"""Standard inertial-frame conversions for orbit initial conditions.

There is no single "Earth-Centered Inertial" frame: the standard realizations
(ICRF/GCRF, EME2000/J2000, TEME, …) differ by frame bias, precession, nutation and
the equation of the equinoxes — an epoch-dependent rotation of up to arc-seconds.
This module pins down *which* realization mjorbit uses and converts a user-supplied
initial state from another named realization into it, using ``astropy`` for the
rigorous, epoch-dependent rotations.

Canonical frame
---------------
mjorbit's canonical orbit frame is **GCRF** (Geocentric Celestial Reference Frame),
whose axes coincide with the IAU ICRS / astropy :class:`~astropy.coordinates.GCRS`
axes. It is the J2000-aligned mean-equator frame already assumed by the bundled solar
ephemeris and environment models (``src/cpp/src/environment.cc``). The canonical time
``t`` is **seconds since the J2000.0 epoch** (2000-01-01 12:00:00 TT, JD 2451545.0),
which is exactly the time argument ``sun_vector_eci`` expects.

``astropy`` is an optional dependency. Conversions are only attempted when an
``OrbitInit`` requests a non-canonical ``frame`` or supplies an absolute ``epoch``;
the default path (``frame="ECI"``, ``epoch=None``) never imports it.
"""

from __future__ import annotations

import numpy as np

#: Name of the inertial realization that bare ``R_eci``/``V_eci`` are interpreted in.
CANONICAL_FRAME = "GCRF"

# JD of the J2000.0 epoch, in the TT scale.
_J2000_JD_TT = 2451545.0

# Frames already aligned with the canonical GCRF axes (identity rotation).
# J2000/EME2000 differ from GCRF only by the constant IAU frame bias (~16 mas,
# i.e. sub-centimetre at LEO), which is far below the fidelity of the bundled
# environment models, so it is neglected here.
_CANONICAL_ALIASES = frozenset({"ECI", "GCRF", "GCRS", "ICRF", "J2000", "EME2000"})

# Frames that require an epoch-dependent rotation into GCRF, performed by astropy.
# TEME (the SGP4/TLE output frame) is inertial-to-inertial relative to GCRF: the net
# rotation rate is only precession/nutation level, so the rotated velocity is recovered
# to ~0.05 mm/s (astropy correctly includes that small frame-rate term and does NOT
# inject Earth rotation).
_ROTATING_FRAMES = frozenset({"TEME"})

# Earth-fixed frames. Positions rotate through astropy's ITRS->GCRS transform; the
# velocity is handled explicitly as v_gcrf = M(t) @ (v_itrf + omega_earth x r_itrf),
# because astropy's matrix transform of a differential would omit the ~0.46 km/s
# omega x r term of a truly Earth-fixed velocity. Neglected: the polar-motion rate
# and LOD variation of |omega| (sub-mm/s at LEO).
_EARTH_FIXED_FRAMES = frozenset({"ITRF", "ITRS", "ECEF"})

# IERS nominal Earth rotation rate (rad/s), the omega of the omega x r term above.
_OMEGA_EARTH_ITRS = 7.292115146706979e-5

#: Input frames accepted by :func:`resolve_orbit_state`.
SUPPORTED_FRAMES = _CANONICAL_ALIASES | _ROTATING_FRAMES | _EARTH_FIXED_FRAMES


def _require_astropy() -> None:
    try:
        import astropy  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via extra-less env
        raise ModuleNotFoundError(
            "mjorbit frame/epoch conversion requires astropy, an optional dependency. "
            "Install it with the 'frames' extra, e.g. `pip install 'mjorbit[frames]'` "
            "or `pixi run -e frames ...`."
        ) from exc


def _as_time(epoch: object):
    """Coerce a user epoch (ISO string / ``datetime`` / ``Time``) to an astropy ``Time``."""
    _require_astropy()
    from astropy.time import Time

    if isinstance(epoch, Time):
        return epoch
    # ISO strings and naive datetimes are interpreted as UTC (astropy's default scale).
    return Time(epoch)


def seconds_since_j2000(epoch: object) -> float:
    """Seconds from the J2000.0 epoch to ``epoch`` (TT), i.e. the canonical ``t``."""
    from astropy.time import Time

    j2000 = Time(_J2000_JD_TT, format="jd", scale="tt")
    return float((_as_time(epoch).tt - j2000).sec)


def _rotate_into_gcrf(
    R_km: np.ndarray, V_kms: np.ndarray, frame: str, epoch_time
) -> tuple[np.ndarray, np.ndarray]:
    _require_astropy()
    from astropy import units as u
    from astropy.coordinates import (
        GCRS,
        TEME,
        CartesianDifferential,
        CartesianRepresentation,
    )

    src_cls = {"TEME": TEME}[frame]
    rep = CartesianRepresentation(
        np.asarray(R_km, dtype=float) * u.km,
        differentials=CartesianDifferential(
            np.asarray(V_kms, dtype=float) * (u.km / u.s)
        ),
    )
    src = src_cls(rep, obstime=epoch_time)
    dst = src.transform_to(GCRS(obstime=epoch_time)).cartesian
    R_out = np.asarray(dst.xyz.to_value(u.km), dtype=float)
    V_out = np.asarray(dst.differentials["s"].d_xyz.to_value(u.km / u.s), dtype=float)
    return R_out, V_out


def _earth_fixed_into_gcrf(
    R_km: np.ndarray, V_kms: np.ndarray, epoch_time
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a truly Earth-fixed (ITRS/ECEF) state into GCRF at ``epoch_time``.

    The position rotates through astropy's rigorous ITRS->GCRS chain (polar motion,
    Earth rotation angle, precession-nutation). The velocity of an Earth-fixed point
    seen from the inertial frame is ``M(t) @ (v_itrs + omega x r_itrs)``; applying the
    rotation matrix alone would drop the ~0.46 km/s Earth-rotation term.
    """
    _require_astropy()
    from astropy import units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

    r_itrs = np.asarray(R_km, dtype=float)
    v_itrs = np.asarray(V_kms, dtype=float)

    # GCRF-from-ITRS rotation matrix at the epoch: transform the ITRS basis vectors
    # (positions only); column j of the result is M @ e_j.
    basis = CartesianRepresentation(np.eye(3) * u.km)
    gcrf_basis = ITRS(basis, obstime=epoch_time).transform_to(GCRS(obstime=epoch_time))
    M = np.asarray(gcrf_basis.cartesian.xyz.to_value(u.km), dtype=float)

    omega = np.array([0.0, 0.0, _OMEGA_EARTH_ITRS])
    R_out = M @ r_itrs
    V_out = M @ (v_itrs + np.cross(omega, r_itrs))
    return R_out, V_out


def resolve_orbit_state(orbit) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve an :class:`~mjorbit.config.OrbitInit` to canonical GCRF state.

    Returns ``(R_eci, V_eci, t)`` where ``R_eci``/``V_eci`` are float ``np.ndarray``
    in km / km·s⁻¹ expressed in the canonical GCRF frame, and ``t`` is seconds since
    J2000.0 when ``orbit.epoch`` is supplied, otherwise ``orbit.t`` unchanged.

    The default (``frame="ECI"``, ``epoch=None``) returns the inputs verbatim and does
    not import astropy.
    """
    frame = str(orbit.frame).upper()
    if frame not in SUPPORTED_FRAMES:
        raise ValueError(
            f"Unsupported orbit frame {orbit.frame!r}. Supported frames: "
            f"{sorted(SUPPORTED_FRAMES)}."
        )

    if frame in _ROTATING_FRAMES or frame in _EARTH_FIXED_FRAMES:
        if orbit.epoch is None:
            raise ValueError(
                f"frame={orbit.frame!r} is epoch-dependent; supply an absolute time via "
                "OrbitInit(epoch=...) (ISO-UTC string, datetime, or astropy Time)."
            )
        if frame in _EARTH_FIXED_FRAMES:
            R_eci, V_eci = _earth_fixed_into_gcrf(
                orbit.R_eci, orbit.V_eci, _as_time(orbit.epoch)
            )
        else:
            R_eci, V_eci = _rotate_into_gcrf(
                orbit.R_eci, orbit.V_eci, frame, _as_time(orbit.epoch)
            )
    else:
        R_eci = np.asarray(orbit.R_eci, dtype=float)
        V_eci = np.asarray(orbit.V_eci, dtype=float)

    t = float(orbit.t) if orbit.epoch is None else seconds_since_j2000(orbit.epoch)
    return R_eci, V_eci, t


__all__ = [
    "CANONICAL_FRAME",
    "SUPPORTED_FRAMES",
    "resolve_orbit_state",
    "seconds_since_j2000",
]
