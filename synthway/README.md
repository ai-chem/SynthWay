# Running SynthWay

Run all commands from the repository root, one level above this folder. For first-time installation, see [setup](../README.md#installation-and-usage).

## Main application

```sh
source .venv/bin/activate
python synthway/synthpath_gui.py
```

Enter a composition (e.g. `Ag`), select the generator and judge, then click **Run**. The application generates synthesis routes and evaluates them; optional regeneration uses the evaluation feedback. Results are saved in a new folder under `outputs/generation/`.

The application and live demo require `OPENROUTER_API_KEY` in `.env`; model calls are billable.

## Other runs

| Purpose | Command | Result |
|---|---|---|
| Live demo | `python scripts/demo.py --out outputs/demo` | One new Ag route and its scores in `outputs/demo/`. Use a new output folder for each run. |
| View a saved example | `python scripts/recorded_example.py` | Saved Ag protocol, scores and laboratory outcome; files in `outputs/recorded_example/`. No API calls. |
| Reproduce article tables | `python paper_tables/analysis/reproduce.py` | 12 tables from saved data; output: `paper_tables/generated/`. No API calls. |
