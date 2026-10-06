# Publication figures V2 — 72 h manuscript

Status: **render-only figure freeze — 6 October 2026**

No scientific experiment was opened for this figure revision. The plots re-express already-frozen/post-freeze evidence for publication layout only. They do not refit a model, recalibrate probabilities, retune thresholds, redesign v3, promote a Phase-H gate, or introduce 24 h / 3 h results.

Reproduction notebook:
- `notebooks/72h_graybox/32_manuscript_publication_figures_v2.ipynb`

## Main figures

1. **Gray-Box trust architecture and chronology.** Makes explicit that the Phase-H physical applicability layer is monitor-only and does not enter v3 routing.
2. **Cross-regime degradation.** Separates TSS, Brier Skill Score, and flare-class conformal coverage into distinct panels.
3. **Reliability paradox.** Separates tightening statistical agreement from falling rare-flare conformal coverage.
4. **Structured AIA failures and graceful degradation.** Shows empirical failure severity alongside NORMAL/DEGRADED routing.
5. **Same-support baseline forest plot.** Reports TSS at validation-chosen/frozen thresholds with active-region 95% intervals and annotates the Cycle-25 paired deltas.
6. **Physical applicability sensitivity.** Shows gradient-family dependence and class-conditioned distance, emphasizing that MD² is not a case-level abstention score.

## Captions

**Figure 1. 72-hour Gray-Box trust architecture and evidence chronology.** Learned SHARP and AIA branches feed a fixed equal-weight fusion. Calibration, Mondrian conformal state and statistical agreement signals determine NORMAL multimodal issuance, while SHARP provides the DEGRADED fallback whenever fusion is not NORMAL but SHARP remains available. The Phase-H physical applicability layer is monitoring-only and does not enter v3 routing. Cycle-25 diagnostics informed the fallback-first semantics without new numerical trust-threshold tuning; supplementary 2026 evaluates frozen v3.

**Figure 2. Cross-regime degradation in discrimination, probabilistic skill and rare-flare conformal transfer.** TSS, Brier Skill Score and flare-class conformal coverage are reported separately for SHARP, AIA and fixed fusion across earlier Cycle-24 development support, Cycle 25 and supplementary 2026.

**Figure 3. Statistical agreement tightens while rare-flare reliability deteriorates.** Panel (a) normalizes fusion seed spread, fusion entropy and SHARP-AIA probability gap to their earlier Cycle-24 values; all decrease in later regimes. Panel (b) shows the simultaneously declining fusion flare-class conformal coverage. The opposing interpretation of these trends motivates treating agreement and reliability as distinct trust dimensions.

**Figure 4. Structured AIA failures and graceful degradation.** Panel (a) shows mapped Gray-Box exposure by empirical AIA failure-episode severity; most mapped cases occur in long clustered episodes. Panel (b) shows that mapped AIA incidents move Cycle-25 forecasts from NORMAL to DEGRADED while issuance remains 1.00 through the SHARP fallback.

**Figure 5. Same-support baseline comparison at validation-chosen/frozen thresholds.** TSS point estimates and 95% active-region block-bootstrap intervals are shown for the post-hoc latest-state logistic comparator and the frozen SHARP, AIA and fusion branches. The bootstrap unit is `region_component_id`, with 2,000 paired draws. Cycle-25 paired deltas are annotated; corresponding 2026 intervals include zero.

**Figure 6. Physical applicability is gradient-dominated and is not a case-level error score.** Panel (a) shows that the aggregate 2026 Mahalanobis-distance elevation disappears when the gradient family, or only the latest gradient axis, is removed. Panel (b) shows that flare windows are closer to the training physical-state cloud than non-flare windows in every regime, arguing against using physical distance as a direct abstention rule.
