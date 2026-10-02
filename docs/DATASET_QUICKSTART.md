# Use the aligned dataset

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

The NumPy-only logistic runner uses the three SHARP histories, fits its transform only on development cases, excludes unknown outcomes and purges unmatured training windows. It uses the 2010–2019 Cycle-24 cohort for development, 2021–2025 for retrospective evaluation and 2026 for supplementary evaluation. These are **exploratory roles**, kept separate from final paper splits. Preparation has been executed; full model fitting is available with `--fit`.

```bash
python scripts/run_exploratory_sharp.py \
  --dataset data/processed/gray_box_aligned_v1 \
  --output-dir outputs/sharp72_first_fit \
  --horizon 72 --scope primary --fit
```

Omit `--fit` for a quick preparation run that writes case roles and counts. Use a fresh output directory each time. Dependencies are the existing `requirements-audit.txt`; no GPU, new downloads or additional libraries are needed. This is a simple logistic reference, not reproduction of the published temporal network or a completed calibrated fusion model.

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
