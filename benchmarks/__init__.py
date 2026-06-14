"""Throughput benchmarks for mjorbit and mjorbit_warp.

Each benchmark builds a small but realistic spacecraft scenario, runs a
fixed number of integration steps on both the CPU rollout path and the
GPU batched-step path, and reports steps/second. Optional comparison
backends (e.g. Basilisk-MuJoCo) are included where they apply.
"""
