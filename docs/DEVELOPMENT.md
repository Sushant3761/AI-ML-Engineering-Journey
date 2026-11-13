# Development guidelines

## Repository layout

- `modules/`: roadmap-aligned learning material. Use one directory per roadmap module, named with its original number and title. Keep projects inside the module they support. Within a module, add only the content areas that are actually needed, such as `notes/`, `implementations/`, `notebooks/`, `experiments/`, and `projects/`.
- `docs/`: guidance that applies across the repository.

Avoid deep nesting that adds no organizational value. Do not create empty content folders in advance or add dummy implementations. A project may link to module notes instead of duplicating them.

## Python and reproducibility

- Prefer a project-specific `pyproject.toml` or a clearly documented requirements file when dependencies are needed.
- State supported Python versions and setup/run commands in the relevant README.
- Keep configuration and secrets out of source control; provide a sanitized `.env.example` only when needed.
- Make random seeds, data sources, and hardware assumptions clear for experiments.
- Keep raw data, trained weights, and generated outputs out of Git unless a small, appropriately licensed sample is essential and documented.

## Notebooks

- Use notebooks for exploration, explanation, and visualization.
- Restart and run notebooks before committing when practical; remove transient outputs that are large or expose local paths or sensitive data.
- Extract reusable code into importable source modules rather than copying it between notebooks.

## Project quality

- A major project should document its goal, scope, setup, usage, tests, and limitations.
- Keep tests close to the project, commonly in `tests/`.
- Prefer explicit, small modules with clear names and avoid coupling unrelated projects.
- Record data provenance and licensing for external datasets and model assets.
