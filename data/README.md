# Data policy

This directory contains documentation only at initialization.

Store large observations, processed tensors and checkpoints in approved external storage. Git may contain small reviewed manifests with source identifiers, object generations/checksums, timestamps, processing versions, quality, coverage and split provenance. Do not commit credentials or private correspondence.

Keep raw, processed, corrected-label and experiment-specific artefacts separately versioned. A new label or preprocessing version must not silently overwrite evidence used for a previous result.

Acquisition continues in [solar-flare-aia-training](https://github.com/Watchman77/solar-flare-aia-training). Record the exact upstream code and data versions when an experiment here consumes those products.
