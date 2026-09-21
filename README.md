# Nordic AI Cup 2026

Workspace for the Nordic AI Cup 2026 (September 17–20, 2026): three independent challenges, each developed as a numbered sequence of experiments with full history kept in `experiments/`.

## Challenges

### `drone/` — Drone Flyby

Realtime object detection and active camera control against a live evaluator (YOLO-based detector + tracker, served over FastAPI).

- `drone/src/` — predictor, tracker, camera control, and model runtime (numbered snapshots, e.g. `predictor_pre_d055.py`, track experiment iterations referenced in the log)
- `drone/scripts/` — dataset mining/building, contact sheets, and validation replay tooling
- Best public validation: `0.1868` (EXP-D058A) · Final hidden evaluation: `0.0181`
- Full write-up: [experiments/drone_experiment_log.md](experiments/drone_experiment_log.md)

### `medical/` — Medical Appointment

Audio-to-text question answering over patient conversations: retrieval + NLI-based classification pipeline served via `medical/api.py`.

- `medical/src/` — retrieval, classification, analysis, deployment, and runtime code, organized by stage
- Best hidden validation: `0.6997` composite (M077 — dual-passage fusion)
- Full write-up: [experiments/medical_experiment_log.md](experiments/medical_experiment_log.md)

### `survival/` — Survival Simulator

Agent-based survival simulation (herbivore hivemind vs. predators, energy/lineage management). See [survival/README.md](survival/README.md) for game rules and observation/action format.

## Repository layout

```
drone/          Challenge 1 — Drone Flyby source, scripts, and generated data (ignored)
medical/        Challenge 2 — Medical Appointment source and FastAPI service
survival/       Challenge 3 — Survival Simulator source, server, and tuned policies
experiments/    Dated experiment logs (EXP-Dxxx / EXP-Mxxx) for drone and medical
scripts/        Environment check + start-up helpers (Windows/Mac)
src/            Shared utilities (e.g. seeding)
data/           Raw/interim/processed data (gitignored, structure only)
requirements-lock.txt, requirements-windows-lock.txt   Pinned environments
```

Generated/large artifacts (model weights, venvs, captures, `runs/`, `medical/artifacts`, `drone/artifacts`) are excluded via `.gitignore` and are not part of this repo.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-windows-lock.txt   # or requirements-lock.txt on macOS/Linux
.venv\Scripts\python scripts\check_env.py                    # verify PyTorch + GPU
```

`scripts/start-windows.ps1` / `scripts/start-mac.sh` pull the latest changes, run the environment check, and open the workspace.

Each challenge exposes its own FastAPI service (`drone/src/server.py`, `medical/api.py`) — see each experiment log's baseline entry for how it's launched and evaluated.
