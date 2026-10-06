# Phase H conclusion — explicit physical/applicability layer

Status: **PHASE H DIAGNOSTIC FREEZE — 6 October 2026**

Phase H added and audited an explicit interpretable magnetic-state layer to the frozen 72 h Gray-Box framework.

## What Phase H established

1. A nonredundant Magnetic Physical State Vector can be constructed from the exact recovered three-slot SHARP tensor using physically interpretable feature families.
2. A training-reference Mahalanobis applicability score detects later distribution changes that are not reflected by decreasing entropy, seed spread or SHARP–AIA disagreement.
3. Physical applicability is not case-level predictive uncertainty: flare windows are often closer to the training physical-state cloud than non-flare windows, and physical distance is negatively associated with fusion squared error.
4. The 2026 applicability elevation is dominated by the latest magnetic-gradient family rather than by a broad shift across all physical families.
5. The apparent April 2026 source-series boundary is not a numerical summary-keyword artefact: paired same-record JSOC queries show exact equality for all 16 SHARP features between `hmi.sharp_720s` and `hmi.sharp_cea_720s`.
6. Therefore physical-state monitoring and statistical confidence should be treated as complementary trust dimensions.

## Frozen claim

> The 72 h Gray-Box framework combines learned forecasts with an explicit interpretable magnetic-state applicability layer. Retrospective analysis shows that statistical agreement can increase while the observed magnetic-feature distribution moves farther from the development reference. The physical signal is gradient-family dominated and is not explained by the SHARP CEA/non-CEA series choice for the 16 summary keywords. Physical applicability is therefore retained as an independent monitoring dimension rather than a direct abstention rule.

## Explicitly not claimed

- no MHD or first-principles physics model;
- no proof that Solar Cycle 25 caused the gradient shift;
- no claim that physical Mahalanobis distance predicts forecast error monotonically;
- no physical gate added to NORMAL / DEGRADED / ABSTAIN;
- no claim of universal predictive superiority.

## Operational interpretation

The Gray-Box trust stack is now:

1. learned SHARP/AIA forecast probabilities;
2. calibration and conformal uncertainty;
3. statistical agreement/disagreement signals;
4. explicit magnetic-state applicability / physical-family diagnostics;
5. data-quality and provenance checks;
6. frozen operational routing.

The physical layer is a monitor that can flag a physically meaningful input-distribution change even when the statistical predictors appear more mutually confident.
