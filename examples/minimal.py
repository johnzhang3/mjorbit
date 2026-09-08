"""Step a free body offset from a circular reference orbit.

Run from the repository root with ``pixi run example-minimal``.
"""

import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import FREE_BODY_XML


def main() -> None:
    radius_km = R_EARTH + 400.0
    orbit = OrbitInit(
        R_eci=[radius_km, 0.0, 0.0],
        V_eci=[0.0, np.sqrt(GM_EARTH / radius_km), 0.0],
    )
    model = MjoModel.from_xml_path(FREE_BODY_XML, mj_timestep=0.01)
    data = model.make_data(orbit=orbit)

    # MuJoCo positions are offsets from the chief, in meters.
    data.qpos[:3] = [10.0, 0.0, 0.0]
    mjo_forward(model, data)
    for _ in range(100):
        mjo_step(model, data)

    print(f"Time: {data.time:.2f} s")
    print("Chief position (km):", data.orbit.R_eci)
    print("Spacecraft offset (m):", data.qpos[:3])


if __name__ == "__main__":
    main()
