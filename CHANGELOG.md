# Changelog

## Unreleased

- Prepare the first public release with Apache 2.0 licensing, software citation
  metadata, contribution guidance, and asset provenance notes.
- Add complete CPU and batched GPU starter examples, a demo catalog, and
  Sphinx documentation for installation, modeling, frames, actuators, sensors,
  planning, and backend synchronization.
- Start the task viewer in free drift by default.
- Clear retained browser transforms when rebuilding viewer scenes so Reset
  keeps articulated arms and solar arrays attached.
- Modernize introductory examples to use editable specs and `model.make_data`.
- Remove the unused CVXPY runtime dependency and exclude local native build
  products from source distributions.
- Add strict docs builds and isolated wheel/source-install checks to CI.
- Publish documentation under `/mjorbit/docs/` alongside the existing project
  page, with combined site previews for PRs and deployment from `main`.
