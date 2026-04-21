# MuJoCo-Orbit Conference Paper Draft

This folder contains a starter IEEE conference paper draft for the MuJoCo-Orbit
project.

## Build

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

or:

```bash
cd paper
make
```

## Draft Structure

- `main.tex`: IEEE conference wrapper, title, abstract, keywords, and section inputs.
- `sections/introduction.tex`: motivation and claimed contributions.
- `sections/related_work.tex`: rough positioning against astrodynamics tools and robotics engines.
- `sections/formulation.tex`: coupled state, LVLH forcing, environmental wrenches, actuators, and stepping.
- `sections/experiments.tex`: planned validation and demonstration matrix.
- `sections/conclusion.tex`: submission plan and current limitations.
- `sections/references.tex`: inline IEEE-style bibliography that compiles on the current TeX install.
- `references.bib`: starter BibTeX scratchpad with TODO notes where citation metadata should be verified.

## Near-Term Paper TODOs

- Replace placeholder author affiliation and email.
- Decide the target conference and check page limit, formatting rules, and anonymity requirements.
- Generate figures and tables from the existing validation tests and examples.
- Verify every bibliography entry against the preferred archival citation.
- Add an explicit limitations paragraph after quantitative experiments are in place.
