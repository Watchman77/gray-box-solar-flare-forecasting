# Data policy

This tracked directory contains documentation. Local raw and processed scientific artefacts live in Git-ignored subdirectories.

Store large observations, processed tensors and checkpoints in approved external storage. Git may contain small reviewed manifests with source identifiers, object generations/checksums, timestamps, processing versions, quality, coverage and split provenance. Do not commit credentials or private correspondence.

Keep raw, processed, corrected-label and experiment-specific artefacts separately versioned. A new label or preprocessing version must not silently overwrite evidence used for a previous result.

Acquisition continues in [solar-flare-aia-training](https://github.com/Watchman77/solar-flare-aia-training). Record the exact upstream code and data versions when an experiment here consumes those products.

The [1 October candidate build](../docs/DATASET_BUILD_20261001.md) stages a pinned canonical metadata object and NOAA annual flare-report files. It references the upstream verified AIA/SHARP arrays rather than duplicating them. Compact provenance, support counts and computational checks are tracked in [results/dataset_construction_20261001](../results/dataset_construction_20261001/). The large per-case outputs remain local; a Git clone alone does not contain them.
