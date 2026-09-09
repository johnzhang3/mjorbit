# mjorbit documentation

mjorbit combines orbital dynamics with MuJoCo multibody simulation for space
robotics. Begin with a free body, then explore articulated spacecraft, contact,
control, and batched GPU simulation.

Read the paper, [*mjorbit: A Simulation Framework for Space Robotics*](https://arxiv.org/abs/2609.08010),
for the methods and evaluation. Citation metadata and BibTeX are available in the
[repository README](https://github.com/johnzhang3/mjorbit#citation-and-license).

- [Install and run the viewer](installation.md).
- [Write your first simulation](quickstart.md).
- [Understand frames and units](frames.md).
- [Configure forces and disturbances](forces-disturbances.md).
- [Run the paper examples](examples.md).

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart
viewer
examples
```

```{toctree}
:maxdepth: 1
:caption: Concepts and guides

architecture
frames
modeling
forces-disturbances
actuators-sensors
planning
gpu
```

```{toctree}
:maxdepth: 1
:caption: Reference and development

api
development
publishing
```

The [project page](https://johnzhang3.github.io/mjorbit/) presents the paper and
visual examples. The source repository contains the experiments and recording
tools used to reproduce those results.
