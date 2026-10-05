"""Build the reviewable, CPU-only earlier-data calibration notebook."""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    config = json.loads((ROOT / "configs/multimodal72_calibration_v1.json").read_text())
    source = (ROOT / "scripts/calibrate_multimodal72.py").read_text()
    cells = []
    def md(value):
        cells.append(nbf.v4.new_markdown_cell(value))
    def code(value):
        cells.append(nbf.v4.new_code_cell(value))
    md("""# 18 · SHARP, AIA and fusion: earlier-data probability calibration

**Bamidele Akinwumi · 72-hour exploratory candidate-label experiment**

The SHARP and AIA three-seed models are already trained. This notebook fits only
probability calibration maps, on the reserved July–December 2014 cases. It can run
on the Mac once that complete role has been archived; it does not wait for later
GPU inference, retrain the neural networks, or score later outcomes.

## Context and methods

- Compare SHARP-only, AIA-only and the prespecified equal-probability average
  (`0.5 * SHARP + 0.5 * AIA`). Fusion weights are fixed, not selected here.
- Use identical valid-input cases for all three branches. Preserve every requested
  row, including failed inputs and their reasons, in the saved calibration table.
- Fit raw, Platt-style and isotonic candidates on July–November 2014, purging any
  outcome plus the assumed 24-hour reporting delay that extends into December.
- Select separately per branch on December 2014 by Brier score, then log loss,
  then declared method order. Refit all candidates on the complete earlier block.
- Reuse the SHARP training-fitted climatology and the tested notebook-02 methods.
  The selected map is saved before later evaluation; no future prediction archive
  enters this notebook.

## Assumptions and limits

These are candidate same-region M/X start labels. Continuous event coverage and
historical delivery are not established. Forecast windows are dependent; the
inner scores below describe method selection, not statistical significance or
independent validation. Conformal uncertainty, operational policy evaluation,
full-population availability and genuinely prospective validation remain separate.

## 1. Configuration and local inputs

Run using the repository training environment from the repository or `notebooks/`.
All source, cohort and parent-prediction hashes are checked. Every calibration
archive must be present and pass its original transfer checks. An incomplete role
stops execution rather than fitting on whichever blocks arrived first.
""")
    code("""from pathlib import Path
from datetime import datetime, timezone
import json, sys
import pandas as pd
from IPython.display import display, Markdown

REPO = Path.cwd().resolve()
if not (REPO / 'scripts/calibrate_multimodal72.py').is_file():
    REPO = REPO.parent
assert (REPO / 'scripts/calibrate_multimodal72.py').is_file(), 'Run inside the repository'
sys.path.insert(0, str(REPO))
__file__ = str(REPO / 'scripts/calibrate_multimodal72.py')
""" + "CONFIG = json.loads(" + repr(json.dumps(config)) + ")\n" + """OUTPUT = REPO / 'outputs' / ('multimodal72_calibration_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
print('New result directory:', OUTPUT)
display(pd.DataFrame([
    {'Stage': 'Inner fitting', 'Period': 'July–November 2014; maturity purge at December boundary'},
    {'Stage': 'Method selection', 'Period': 'December 2014'},
    {'Stage': 'Final refit', 'Period': 'Complete eligible July–December 2014'},
    {'Stage': 'Later evaluation', 'Period': 'Not performed in this notebook'}
]))""")
    md("## 2. Executable methods\n\nThe code below is the exact tested calibration implementation. Its dependency hashes are pinned in the configuration.")
    code(source)
    md("## 3. Verify the complete earlier block, select and save the maps")
    code("""summary = run(REPO, CONFIG, OUTPUT)
selection = json.loads((OUTPUT / 'selection.json').read_text())
print(summary['status'])
print('Later evaluation roles read:', summary['evaluation_roles_read'])
print('GPU calls:', summary['GPU_calls'])""")
    md("## 4. Matched support and explicit exclusions")
    code("""display(pd.DataFrame(summary['matched_support']).T)
display(pd.DataFrame([{'Requested calibration cases': summary['requested_cases'],
                       'Input failures preserved': summary['input_failure_cases']}]))""")
    md("## 5. Earlier method-selection evidence\n\nThese scores determine the choice. They do not establish which branch performs best in Cycle 25 or 2026.")
    code("""scores = pd.DataFrame(selection['inner_scores'])
scores['selected'] = [method == selection['selected_methods'][branch]
                      for branch, method in zip(scores.branch, scores.method)]
display(scores[['branch', 'method', 'cases', 'positive', 'brier', 'log_loss', 'selected']].round(6))
display(pd.DataFrame([{'Branch': branch, 'Selected map': method}
                      for branch, method in selection['selected_methods'].items()]))""")
    md("## 6. Saved outputs and next stage\n\nUse the frozen maps on the reserved uncertainty-calibration block, then evaluate the locked methods on matched later cases. No fusion improvement or operational readiness is inferred here.")
    code("""display(Markdown('Saved calibrators: `' + str(OUTPUT / 'selection.json') + '`'))
print('Cases:', OUTPUT / 'calibration_cases.csv.gz')
print('Source receipts:', OUTPUT / 'source_receipt.json')
print('Status: earlier-data calibration complete; later evaluation remains pending.')""")
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbf.validate(nb)
    destination = ROOT / "notebooks/18_SHARP_AIA_Fusion_72h_Calibration.ipynb"
    nbf.write(nb, destination)
    return destination


if __name__ == "__main__":
    print(build())
