"""Step 256 worlds with explicit GPU upload and pull.

Run on Linux with ``pixi run -e warp example-batched``.
"""

import numpy as np

from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import FREE_BODY_XML
from mjorbit_warp import MjoModel, OrbitInit, mjo_forward, mjo_pull, mjo_step, mjo_upload


def main() -> None:
    radius_km = R_EARTH + 400.0
    model = MjoModel.from_xml_path(FREE_BODY_XML, mj_timestep=0.01)
    data = model.make_data(
        orbit=OrbitInit(
            R_eci=[radius_km, 0.0, 0.0],
            V_eci=[0.0, np.sqrt(GM_EARTH / radius_km), 0.0],
        ),
        nworld=256,
    )

    # Each world starts at a different chief-relative offset, in meters.
    data.qpos[:, 0] = np.linspace(0.0, 10.0, data.nworld)
    mjo_upload(model, data, fields="qpos")
    mjo_forward(model, data)
    for _ in range(100):
        mjo_step(model, data)
    mjo_pull(model, data, fields=["time", "qpos", "orbit"])

    print("qpos shape:", data.qpos.shape)
    print("Time (s):", data.time)
    print("First/last spacecraft offsets (m):", data.qpos[[0, -1], :3])


if __name__ == "__main__":
    main()
