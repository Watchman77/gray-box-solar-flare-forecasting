# Literature positioning audit — 72 h Gray-Box manuscript

Status: **working literature map — 6 October 2026**

This note records the literature basis used to position the manuscript. It is intentionally conservative: it does not claim that calibration, uncertainty, multimodal flare forecasting, operational verification, temporal evaluation, or conformal prediction are individually novel.

## 1. Forecast baselines and probabilistic verification

- **Wheatland (2005), Space Weather, 3, S07003, doi:10.1029/2004SW000131.** Developed a Bayesian event-statistics flare forecast and explicitly argued that simple statistical forecasts should serve as baselines that richer methods must improve upon.
- **Crown (2012), Space Weather, doi:10.1029/2011SW000760.** Validated NOAA/SWPC probabilistic flare forecasts using Brier Skill Score and contingency-table measures.
- **Camporeale & Berger (2025), Space Weather, 23, e2025SW004546, doi:10.1029/2025SW004546.** Verified NOAA/SWPC M/X flare forecasts over 1998–2024 and found that simple persistence, climatology and lightweight statistical baselines can match or outperform the operational forecasts on important metrics; calibration and false alarms remain major issues.

**Positioning implication:** our manuscript must include climatology and a simple logistic comparator and must report probabilistic as well as categorical verification. Baselines are not optional.

## 2. Magnetic-field machine learning and temporal modelling

- **Bobra & Couvidat (2015), ApJ, 798, 135, doi:10.1088/0004-637X/798/2/135.** Established large-scale vector-magnetogram ML flare prediction using SDO/HMI SHARP parameters and emphasized TSS and feature selection.
- **Liu et al. (2019), ApJ, 877, 121, doi:10.3847/1538-4357/ab1b3c.** Introduced LSTM temporal modelling for flare prediction using magnetic and flare-history time series.
- **Angryk et al. (2020), Scientific Data, 7, 227, doi:10.1038/s41597-020-0548-x.** Released the SWAN-SF multivariate SHARP time-series benchmark and stressed consistent datasets, labeling and comparison.
- **Akinwumi et al. (2026), Advances in Space Research, doi:10.1016/j.asr.2026.09.024.** Previous work from the present research programme developed evolution-aware temporal SHARP modelling across multiple horizons. The current manuscript must not recycle that contribution as the main novelty.

**Positioning implication:** temporal SHARP modelling is established. The new paper must be about operational trust around frozen predictors, not the existence of the temporal model itself.

## 3. Ensemble and multimodal forecasting

- **Guerra et al. (2015), Space Weather, 13, 626–642, doi:10.1002/2015SW001195.** Demonstrated ensemble combinations of multiple flare forecasts and showed that gains can depend on threshold and member biases.
- **Francisco et al. (2024), “Multimodal Flare Forecasting with Deep Learning”, arXiv:2410.16116.** Compared magnetogram, UV/EUV and multimodal deep-learning inputs and reported complementarity across atmospheric layers.
- **Riggi et al. (2026), Astronomy and Computing, 55, 101042, doi:10.1016/j.ascom.2025.101042.** Compared foundation-model approaches across image, video and time-series modalities, reinforcing the active development of multimodal and temporal flare forecasting.
- **ARCAFF (EU Horizon 2022–2026, grant 101082164).** Explicitly targeted end-to-end deep-learning flare prediction, multimodal time-series products and forecast uncertainties.

**Positioning implication:** multimodality, ensembles and forecast uncertainty are not individually novel. Our equal-weight fusion is a fixed branch used to study trust and degradation, not a claim of novel fusion superiority.

## 4. Operational comparison and research-to-operations

- **Barnes et al. (2016), ApJ, 829, 89.** “All-Clear” comparison showed that consistent data and metrics are essential and that no one method clearly dominates.
- **Leka et al. (2019), ApJS, 243, 36, doi:10.3847/1538-4365/ab2e12.** Compared operational flare forecasting systems with common benchmarks and metrics; no single winner emerged across event definitions and metrics.
- **Leka et al. (2019), ApJ, 881, 101, doi:10.3847/1538-4357/ab2e11.** Examined systematic behaviours of operational systems and highlighted prior flare activity, region evolution and operational implementation effects.
- **Georgoulis et al. (2021), Journal of Space Weather and Space Climate, 11, 39, doi:10.1051/swsc/2021023.** FLARECAST had an explicit research-to-operations focus and emphasized rigorous training/testing and open infrastructure.
- **Nishizuka et al. (2021), Earth, Planets and Space, 73, 64, doi:10.1186/s40623-021-01381-9.** DeFN provides a strong example of chronological and genuinely operational flare-forecast evaluation.

**Positioning implication:** the manuscript must call its replay “prospective-style chronological replay,” not live prospective deployment.

## 5. Temporal shift and solar-cycle effects

- **Goodwin, Sadykov & Martens (2024), ApJ, 964, 163, doi:10.3847/1538-4357/ad276c.** Simulated real-time flare forecasting with stationary, rolling and expanding training windows and demonstrated that solar-cycle conditions affect forecast behaviour.

**Positioning implication:** temporal robustness and solar-cycle effects are already recognized. Our contribution is the joint evaluation of calibration, conformal reliability, agreement signals, modality failures and trust routing under later temporal regimes.

## 6. Uncertainty and conformal prediction

- **Hong, Pandey & Aydin (2026), “Uncertainty-Aware Solar Flare Regression”, arXiv:2603.06712.** Applies conformal prediction, quantile regression and conformalized quantile regression to solar-flare regression.
- **ARCAFF** also explicitly planned forecast uncertainties as an output.

**Positioning implication:** do not claim that conformal prediction or uncertainty-aware flare forecasting is new. Our binary Mondrian conformal analysis is one component of a broader operational trust experiment, and later-regime coverage degradation is an empirical result rather than a universal conformal guarantee.

## 7. Defensible novelty statement

The literature reviewed above treats many necessary components separately: magnetic and temporal prediction, multimodality, ensemble combination, calibration and verification, uncertainty quantification, chronological evaluation and operational benchmarking.

The present manuscript is therefore positioned around **integration and experimentally frozen decision behaviour**, not invention of any one component:

> a 72-hour active-region Gray-Box trust framework that combines calibrated frozen predictors, conformal uncertainty, model-agreement diagnostics, interpretable magnetic-state applicability, provenance/data-quality evidence, realistic AIA acquisition failures, and explicit NORMAL / DEGRADED / ABSTAIN routing, then evaluates how those components behave together under later temporal shift.

This is a positioning statement, not a “first ever” claim. Any stronger priority claim should be made only after a dedicated systematic literature search.

## 8. Claims to avoid

- “first uncertainty-aware solar flare forecast”
- “first conformal solar-flare model”
- “first multimodal solar-flare forecast”
- “first operational flare-forecast evaluation”
- “first solar-cycle robustness study”
- “fusion improves forecasting”
- “physical OOD predicts errors”
- “2026 gradient shift is caused by Solar Cycle 25”
- “prospective validation” without the qualifier “prospective-style chronological replay”
