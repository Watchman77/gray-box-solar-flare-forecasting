# Gray-Box AIA72 Results Synthesis

## Frozen experimental chronology

1. **Marginal conformal v1** failed operationally because high marginal coverage masked rare-flare undercoverage and the predeclared q90 policy abstained on too many flare cases.
2. **Mondrian/class-conditional conformal v2** repaired class balance on the calibration block but did not preserve nominal flare-class coverage under later temporal shift.
3. The **Cycle 25 diagnostic** showed that using uncertainty as a hard abstention trigger disproportionately withheld flare-producing cases.
4. **Fallback-first v3** was therefore frozen before the supplementary-2026 evaluation: trusted fusion is NORMAL, SHARP fallback is DEGRADED, and ABSTAIN is reserved for genuine fallback unavailability.
5. The **supplementary-2026 evaluation** remained one-shot and locked; subsequent outage tests are explicitly post-hoc robustness analyses.

## Clean supplementary-2026 result

On 11,118 cases with 992 positives:

- Fusion: TSS 0.292, recall 0.340, specificity 0.952.
- Fallback-first v3 q90: TSS 0.296, recall 0.346, specificity 0.951.
- SHARP: TSS 0.323, recall 0.424, specificity 0.898.

Relative to fusion, v3 q90 changed TSS by 0.0042 with 95% bootstrap CI
[-0.0003, 0.0095]. The CI includes zero, so the evidence does not support a strong TSS-superiority claim.
The exact paired McNemar p-value versus fusion was 0.06599.

Relative to SHARP, v3 q90 changed TSS by -0.0264 with 95% bootstrap CI
[-0.0439, -0.0095]. SHARP therefore retained the stronger TSS/recall operating point,
while v3 provided substantially higher specificity/precision and fewer false alarms.

## Conformal result

The central uncertainty finding is not that conformal prediction failed universally, but that guarantees calibrated on earlier data did not remain class-wise reliable under later temporal shift.
Rare-flare coverage degraded much more strongly than no-flare coverage, demonstrating why aggregate coverage alone is inadequate for this task.

## Operational robustness

Detected AIA loss can be routed to SHARP fallback, preserving forecast availability and flare retention. As AIA missingness increases,
fusion-only availability declines while v3 remains available so long as SHARP is healthy.

When SHARP itself fails, the frozen architecture loses both the multimodal fusion route and the fallback route.
In that case ABSTAIN becomes unavoidable. This provides empirical justification for the three operational states:

- **NORMAL:** trusted multimodal fusion.
- **DEGRADED:** SHARP fallback.
- **ABSTAIN:** no valid fallback path.

## Reviewer-safe contribution statement

The main contribution is not a claim of universal predictive superiority. It is an experimentally validated operational trust layer
that exposes when uncertainty-based abstention is unsafe for rare events, preserves service through modality-aware fallback,
and makes unavoidable abstention explicit when the fallback itself is unavailable.
