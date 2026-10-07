# GOES Multi Horizon Protocol

## Current status

The aligned Gray-Box package contains GOES-related event records used for outcome construction. It does not yet contain a verified continuous pre-issue GOES X-ray predictor matrix. GOES event labels must never be reused as model inputs.

## Two distinct uses of GOES

### GOES as the outcome source

The event catalogue identifies reported flare timing and class. For each issue time and horizon, the label is constructed from the event interval. This is the current role and is separate from model features.

### GOES as a predictor branch

A future GOES branch would use only measurements available at or before issue time, for example recent X-ray flux, background level, derivative, channel availability, and quality flags. It would require:

- a versioned time-series acquisition receipt;
- explicit issue-time cutoffs and cadence resampling;
- no measurements from the forecast interval;
- channel and instrument quality masks;
- normalization fitted on the training role only;
- missingness and outage replay;
- a separate GOES-only model and matched SHARP/GOES/fusion support.

## Recommended sequence

1. Train and evaluate horizon-specific SHARP models first.
2. Add AIA only when the corresponding image support and GPU run are available.
3. Build and audit a continuous GOES predictor matrix separately.
4. Train GOES-only and SHARP+GOES branches on identical cases.
5. Add SHARP+AIA+GOES fusion only after the single-branch inputs pass the same leakage and availability gates.

This keeps GOES event information available for labels while preventing accidental target leakage into a predictor branch.
