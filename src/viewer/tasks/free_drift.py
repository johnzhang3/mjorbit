# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Free drift: dual-arm spacecraft tumbling gently in LEO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.testdata import SPACECRAFT_DUAL_ARM_PANELS_XML

from .base import ViewerTask, circular_orbit_eci, ui_field
from .registry import register_task


@dataclass
class FreeDriftParams:
    altitude_km: float = ui_field(500.0, 250.0, 2000.0, step=10.0)
    inclination_deg: float = ui_field(51.6, 0.0, 98.0, step=0.1)
    tumble_deg_s: float = ui_field(0.5, 0.0, 5.0, step=0.05)
    arm_slew_amplitude: float = ui_field(0.6, 0.0, 1.2, step=0.05)
    arm_slew_period_s: float = ui_field(45.0, 5.0, 240.0, step=1.0)


@register_task
class FreeDriftTask(ViewerTask):
    """Uncontrolled bus with two arms and solar wings on a circular orbit."""

    name = "free_drift"
    description = (
        "Dual-arm spacecraft with solar wings drifting on a circular LEO "
        "orbit: slow tumble plus a gentle periodic arm slew. Altitude, "
        "inclination, and tumble rate apply on Reset."
    )
    track_body = "bus"
    trail_bodies = ("bus",)

    params: FreeDriftParams

    def make_params(self) -> FreeDriftParams:
        return FreeDriftParams()

    def build(self) -> tuple[MjoModel, MjoData]:
        R_eci, V_eci = circular_orbit_eci(self.params.altitude_km, self.params.inclination_deg)
        model = MjoModel.from_xml_path(SPACECRAFT_DUAL_ARM_PANELS_XML)
        data = model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
        rate = np.deg2rad(self.params.tumble_deg_s)
        data.qvel[3:6] = rate * np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])
        mjo_forward(model, data)
        return model, data

    def pre_step(self) -> None:
        t = float(self.data.time)
        # Soft-start so the position servos don't kick the free base at t=0.
        amplitude = self.params.arm_slew_amplitude * min(t / 10.0, 1.0)
        omega = 2.0 * np.pi / max(self.params.arm_slew_period_s, 1.0e-3)
        phase = omega * t
        self.data.ctrl[:4] = amplitude * np.array(
            [
                np.sin(phase),
                -0.6 * np.cos(phase),
                np.sin(phase + np.pi / 3.0),
                -0.6 * np.cos(phase + np.pi / 3.0),
            ]
        )

    def status(self) -> str:
        omega_deg = np.rad2deg(np.linalg.norm(np.asarray(self.data.qvel[3:6])))
        altitude = np.linalg.norm(self.data.orbit.R_eci) - R_EARTH
        return f"**body rate** = {omega_deg:.2f} deg/s &nbsp; **altitude** = {altitude:.0f} km"
