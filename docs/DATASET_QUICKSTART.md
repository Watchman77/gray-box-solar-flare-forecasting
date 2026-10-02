# Use the aligned dataset

For model training, open [01_SHARP_72h_Training.ipynb](../notebooks/01_SHARP_72h_Training.ipynb). It is the primary training interface and contains executable training code and saved outputs. Install `requirements-notebook.txt` into the selected kernel if needed. The notebook uses the verified archive or `data/processed/training_snapshot_20261002/gray_box_aligned_v1/`: this separate snapshot preserves the working package after its derived `dataset.csv` changed. The arrays and labels were unchanged, and the original archive checksum was verified before extraction.

Continue with [02_SHARP_72h_Calibration.ipynb](../notebooks/02_SHARP_72h_Calibration.ipynb) for earlier-data calibration and method selection. Its configuration pins the exact training output and dataset; another training run requires a separately pinned experiment configuration. It keeps the backbones fixed and preserves the conformal/policy blocks.

Continue with [03_SHARP_72h_Conformal_Uncertainty.ipynb](../notebooks/03_SHARP_72h_Conformal_Uncertainty.ipynb) for pooled and class-conditional sets fitted on January–June 2015. It pins the Notebook 02 output, reports class-specific and annual coverage at 90%/95% targets, and preserves missing-input/unknown-outcome cases. Its recorded transfer failures are retained; the operational policy is not yet validated.

Continue with [04_SHARP_72h_Rolling_Conformal.ipynb](../notebooks/04_SHARP_72h_Rolling_Conformal.ipynb) for daily uncertainty updates using only matured earlier outcomes. Its configuration pins the exact Notebook 02 probabilities and Notebook 03 fixed class-conditional thresholds. It compares 90/180/365-day windows, selects a method on July 2015–2019, and records later coverage alongside ambiguous sets and insufficient-support guards. This stage consumes the earlier policy-development block; that block is no longer untouched. The notebook has executed with saved tables and three figures. Full per-case sets and daily threshold journals remain local; [compact results and verification](../results/sharp72_rolling_20261002/) are versioned.

Continue with [05_SHARP_72h_Decision_Policy.ipynb](../notebooks/05_SHARP_72h_Decision_Policy.ipynb) for experimental state routing and abstention. It pins the original training outputs and Notebook 04 sets, fits a feature-distance reference on 2010–2013, and freezes distance/disagreement limits from January–June 2014. The executed comparisons expose the logistic fallback's high error rate and the guarded GRU's large flare-deferral fraction. State names do not certify safety. Per-case decisions remain local; [compact outputs and verification](../results/sharp72_policy_20261002/) are versioned. No new AIA download or backbone training is required.

The assembled dataset is in `data/processed/gray_box_aligned_v1/`. The portable archive is `data/processed/gray_box_aligned_v1.zip`. Large files are local and Git-ignored; the committed [manifest](../results/dataset_package_20261002/manifest.json) records their contents and checksums.

| Open this | Contents |
|---|---|
| `dataset.csv` | Full readable table: 153,366 candidate records, 104 columns, including all 48 SHARP values (16 parameters × three history slots), AIA references and 48/72-hour outcomes. |
| `preview.csv` | 178 representative rows from different years, input states and outcome states. This is a viewing sample, not a prevalence sample. |
| `data_dictionary.csv` | Column meanings and unknown-value conventions. |
| `sharp.npy` | Raw float64 tensor, shape `(113433, 3, 16)`. |
| `sharp_missing.npy` | Feature-missingness mask with the same shape; all assembled feature values are finite. |
| `input_index.csv.gz` | Exact tensor-row → case → three existing AIA object paths. |
| `y_primary_72h.npy` | Example target array: 1 = candidate event, 0 = provisional no-event, −1 = unknown. |
| `known_primary_72h.npy` | True only where the corresponding candidate label is 0 or 1. |
| `goes_events.csv.gz` | Versioned GOES event catalogue used for outcomes. |
| `master_cases.csv.gz` | Compact case table, preserving all 39,933 records without assembled matched inputs. |

Both primary-NOAA and current-HARP NOAA-union targets are available at 48 and 72 hours. The three history slots are issue minus 288, 192 and 96 native minutes. AIA pixels stay in the existing GCS bucket; this package does not repeat that download. GOES here supplies event labels, not a continuous XRS input branch.

Blank target cells in the CSV and −1 in the arrays mean **unknown**, never “no flare.” Blank SHARP values on unmatched cases mean **inputs not assembled**. `tensor_row=-1` is a missing pointer, not a Python index to use. Existing upstream 48-hour labels are retained in their own column.

## View without AIA

`data/processed/gray_box_sharp_goes_view_v1/sharp_goes.csv` contains 153,366 rows × 73 columns, with all three SHARP histories, region identifiers, observation times, and explicit primary/patch 48/72-hour outcomes. It has no AIA paths or pixels. The adjacent `sample.csv` has 178 viewing rows and shows only the most recent SHARP slot (issue minus 96 native minutes). The [export receipt](../results/pipeline_validation_20261002/sharp_goes_view_manifest.json) pins the source and files; every exported cell was compared after CSV readback.

This is a view of the existing aligned cohort, not an independently expanded SHARP-only population. It retains 39,933 cases with unassembled SHARP histories as blank values. GOES provides event labels here, not continuous X-ray flux predictors. The uploaded `SOLAR_FLARE_FULL_2010_2025.csv` is preserved as a reference: its unspecified `FLARE_BINARY` horizon is not assumed equivalent to either new target. Its feature set contains `AREA_ACR`, whereas the current 16-feature set contains `MEANJZH` instead.

```bash
python scripts/export_sharp_goes.py \
  --dataset data/processed/gray_box_aligned_v1 \
  --output-dir data/processed/gray_box_sharp_goes_view_v1
```

## Read it

The CSV can be opened directly in a spreadsheet application. For modelling, use the arrays and index to avoid reading the large text export:

```python
from scripts.dataset_io import load_dataset

data = load_dataset("data/processed/gray_box_aligned_v1", horizon=72, scope="primary")
X = data["sharp"]
y = data["labels"]
known = data["known"]
index = data["index"]

# Select a declared development period as well as the outcome mask.
# Do not pass -1 labels to a binary classifier.
```

After extracting the portable archive, its included `dataset_io.py` provides the same reader without requiring this repository.

## Run the CPU baseline

The NumPy-only logistic runner uses the three SHARP histories, fits its transform only on development cases, excludes unknown outcomes and purges unmatured training windows. It uses the 2010–2019 Cycle-24 cohort for development, 2021–2025 for retrospective evaluation and 2026 for supplementary evaluation. These are **exploratory roles**, kept separate from final paper splits. Both 48-hour and 72-hour fits have now completed; see [pipeline fixes and results](PIPELINE_STATUS.md).

```bash
python scripts/run_exploratory_sharp.py \
  --dataset data/processed/gray_box_aligned_v1 \
  --output-dir outputs/sharp72_first_fit \
  --horizon 72 --scope primary --fit
```

Omit `--fit` for a quick preparation run that writes case roles and counts. Use a fresh output directory each time. Dependencies are the existing `requirements-audit.txt`; no GPU, new downloads or additional libraries are needed. This is a simple logistic reference, not reproduction of the published temporal network or a completed calibrated fusion model.

The current repository reader enables checksums by default and the runner validates the full package before fitting. Add `--resume` to reuse a completed result only if its data/code/target contract and saved hashes still match. For portable use, the updated `gray_box_pipeline_tools_v1.zip` contains these safeguards; the original scientific-data archive remains unchanged. Its older embedded reader is retained only as the original package snapshot.

## Rebuild from cached sources

```bash
python scripts/package_model_dataset.py \
  --aia-root "/path/to/AIA Solar flare Project" \
  --inventory-dir data/processed/gray_box_inventory_v2 \
  --outcome-dir data/processed/gray_box_outcomes_v2 \
  --output-dir data/processed/gray_box_aligned_v1

python scripts/export_dataset_table.py --dataset data/processed/gray_box_aligned_v1
```

All labels remain science-class, start-time candidates. Continuous outcome coverage and historical delivery have not been established, and previously inspected years are not untouched prospective confirmation. The package preserves those flags so modelling preparation can proceed without hiding the limits or requiring every ambiguous event to be resolved first.
