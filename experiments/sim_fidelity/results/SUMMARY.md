# Sim-fidelity study — results

_Training backend on the x-axis; every policy evaluated in the full mjorbit_warp orbital environment._

## Truss (nadir-pointing)

Success = hug retained for the whole episode **and** final-quarter mean nadir error below the threshold. Eval in full mjorbit_warp, identical worlds across all policies.

| condition | seeds | alive | lost | final err (deg) | succ@3° | succ@5° | succ@10° |
|---|---|---|---|---|---|---|---|
| mjorbit-warp (full orbital) | 3 |  95.1 ±  0.4% |   4.9 ±  0.4% |  1.501 ± 0.202 |  94.3 ±  0.2% |  94.5 ±  0.0% |  94.6 ±  0.2% |
| mjwarp fair (bare dyn, moving nadir) | 3 |  94.0 ±  0.9% |   6.0 ±  0.9% |  1.945 ± 0.419 |  80.1 ±  8.5% |  90.3 ±  3.8% |  92.8 ±  2.5% |
| mjwarp naive (bare dyn, frozen nadir) | 3 |  94.9 ±  0.4% |   5.1 ±  0.4% |  1.687 ± 0.543 |  91.1 ±  4.2% |  94.3 ±  0.1% |  94.5 ±  0.0% |

## Astrobee (detumble + grasp) — predicted-null control

Success = grasped+held for the final quarter, no fail/crash. No orbital-frame reference in this task, so mjwarp and mjorbit should be close.

| condition | seeds | success | grasped frac | final spin (rad/s) | failed |
|---|---|---|---|---|---|
| mjorbit-warp (full orbital) | 3 |  77.9 ± 11.3% |  76.2 ±  9.9% |  0.274 ± 0.046 |   8.1 ±  4.4% |
| mjwarp (bare dynamics) | 3 |  70.4 ± 20.7% |  71.8 ± 20.0% |  0.228 ± 0.120 |  10.5 ± 10.2% |
