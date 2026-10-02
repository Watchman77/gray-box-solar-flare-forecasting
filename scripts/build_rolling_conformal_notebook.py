"""Create notebook 04 with tested past-only rolling replay code embedded."""
import ast
import json
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]


def build():
    def read_functions(path):
        source=path.read_text()
        return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    shared=read_functions(ROOT/'scripts/conformal_sharp.py')
    rolling=read_functions(ROOT/'scripts/rolling_conformal_sharp.py')
    config=json.loads((ROOT/'configs/sharp72_rolling_conformal_v1.json').read_text())
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md('''# 72-hour SHARP rolling uncertainty replay
**Notebook 04 · Bamidele Akinwumi · exploratory candidate-label experiment**

## tl;dr
Test whether daily updates using matured past outcomes improve the fixed-2015 class-conditional baseline. The saved execution below reports coverage, ambiguity, support guards and annual behavior. No backbone or probability mapper is retrained.''')
    md('''## Context & Methods

Notebook 03 exposed poor transfer of fixed conformal thresholds. This experiment compares those exact **fixed class-conditional thresholds** with **90-, 180- and 365-day rolling class-conditional thresholds**. Pooled prediction is retained in Notebook 03; this notebook focuses on class coverage.

At **00:00 UTC each day**, select calibration windows issued in the preceding lookback interval whose complete 72-hour outcome window plus an assumed **24-hour reporting delay** has elapsed. Apply the resulting thresholds to that day's inherited issue-time cases. Later outcomes cannot alter earlier sets. Both classes use the same delayed feedback rule, including positive outcomes.

Choose one method per backbone on **July 2015–2019 development outcomes**, at the primary 90% target: minimize the worst class coverage deficit below 90%, then mean set size, then declared candidate order. The fixed baseline remains a candidate. Reuse that choice at the 95% sensitivity. Save the choice before scoring later periods; report all declared candidates without changing the choice.

### Key Assumptions

- Outcomes are provisional M/X start-within-72-hour labels for the primary NOAA region. Counts refer to overlapping forecast windows.
- Models and probability mappings remain frozen. Rolling calibration does use matured earlier Cycle-25/2026 labels: this tests sequential updating, not label-free cycle transfer.
- A class needs at least **50 windows from five mapped region components** in the lookback. Otherwise include that class automatically. These are declared support heuristics, not a coverage theorem. Both labels may be included when support is weak; this is reported explicitly.
- Missing SHARP inputs receive no set; unknown outcomes receive sets when inputs exist but never enter calibration or metrics. No 2020 scored cases are available, so 2021 starts without recent observations.
- Historical reporting/revision times are unverified. The 24-hour delay is an assumption, and this is a retrospective replay on native cases rather than a complete prospective forecast schedule.
- This implements rolling quantiles at fixed alpha. It does **not** implement an adaptive-conformal error-feedback controller or claim its guarantees. Temporal dependence and drift can still break nominal coverage.
- The experiment was designed after inspecting earlier retrospective results; it is exploratory. The 2015–2019 block is now used for rolling-method development and cannot be presented as untouched validation for later policy claims.
- Bootstrap intervals are conditional paired resampling of already-issued sets. They do not rerun the sequential updates and exclude model, calibration-sample, label and method-selection uncertainty.

Related primary sources: [Barber et al., Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415) explains why changing distributions matter; [Gibbs & Candès, Adaptive Conformal Inference](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) describes a distinct adaptive method for later comparison. This notebook's windows, support guards and selection rule are experimental choices, not implementations of those papers' theorems.''')
    md('## Data\n### 1. Environment and pinned inputs\nUse the same local training kernel and `requirements-notebook.txt`; no new dependencies, cloud jobs or downloads are needed. Functions are embedded so no repository Python imports are required.')
    imports='''from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd'''
    code(imports+'''
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Markdown
WORKSPACE=Path.cwd()
if WORKSPACE.name=='notebooks':WORKSPACE=WORKSPACE.parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':115,'axes.titleweight':'bold'})
COLORS={'fixed_2015':'#245A81','selected':'#B36B19'}
PERIOD_NAMES={'retrospective_cycle25':'2021–2025','supplementary_2026':'2026 (partial)'}
print({'numpy':np.__version__,'pandas':pd.__version__})''')
    code("CONFIG=json.loads(r'''"+json.dumps(config,indent=2)+"''')\n"+'''PARENT_DIR=WORKSPACE/CONFIG['parent_dir']
FIXED_DIR=WORKSPACE/CONFIG['fixed_dir']
DATASET_DIR=WORKSPACE/'data/processed/training_snapshot_20261002/gray_box_aligned_v1'
OUTPUT_DIR=WORKSPACE/'outputs'/('sharp72_rolling_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
print('New run:',OUTPUT_DIR)
display(pd.DataFrame([{'Stage':'Threshold update','Rule':CONFIG['window_membership']},
    {'Stage':'Method selection','Rule':CONFIG['selection']},
    {'Stage':'Low support','Rule':CONFIG['insufficient_support']}]))''')
    md('### 2. Reuse the verified conformal definitions and source checks\nExact finite-sample ranks, inclusive ties and set encoding follow Notebook 03. A blank set is distinct from an empty set.')
    code('\n\n'.join(shared[n] for n in ['sha256','save_json','validate_probabilities','finite_sample_threshold','predict_sets','set_metrics','coverage_bootstrap','load_sources']))
    md('### 3. Enforce the daily information boundary\nLookback membership is based on issue time. Outcome availability is checked separately, preventing the latest unresolved 72-hour windows from leaking into calibration. Insufficient support always includes the affected class.')
    code('\n\n'.join(rolling[n] for n in ['code_column','decode_sets','prepare_history','eligible_history','daily_thresholds']))
    md('### 4. Issue sets and record every threshold update\nEach journal row records the cutoff, class, rank, support, latest assumed availability and a hash of included case identities. The journal and full population sets stay local.')
    code(rolling['generate_replay'])
    md('### 5. Select only on earlier development outcomes\nCoverage deficit is zero only if both observed class coverages reach the nominal target. Mean set size then rewards more informative sets. This selection rule uses point estimates and is not a safety certificate.')
    code(rolling['select_on_policy'])
    md('### 6. Evaluate and compare paired forecasts\nFor selected minus fixed differences, positive coverage differences mean greater inclusion of the correct outcome; positive set-size or both-label differences mean less specific sets. Region components and seven-day UTC blocks provide conditional resampling sensitivities.')
    code('\n\n'.join(rolling[n] for n in ['paired_set_bootstrap','evaluate_replay','run_rolling']))
    md('### 7. Run the chronological replay\nAll candidate settings are recorded before replay. The method-selection function accesses only the earlier development labels, and its result is saved before later evaluation.')
    code("SOURCE_IMPORTS="+repr(imports)+'\n'+'''history=get_ipython().history_manager.input_hist_raw
starts=['def sha256','def code_column','def generate_replay','def select_on_policy','def paired_set_bootstrap']
EXECUTED_SOURCE=SOURCE_IMPORTS+'\\n\\n'+'\\n\\n'.join(next(c for c in reversed(history) if c.startswith(start)) for start in starts)
summary=run_rolling(PARENT_DIR,FIXED_DIR,DATASET_DIR,CONFIG,OUTPUT_DIR,EXECUTED_SOURCE)
selection=json.loads((OUTPUT_DIR/'selection.json').read_text())
metrics=pd.read_csv(OUTPUT_DIR/'metrics.csv')
yearly=pd.read_csv(OUTPUT_DIR/'yearly_metrics.csv')
monthly=pd.read_csv(OUTPUT_DIR/'monthly_metrics.csv')
intervals=pd.read_csv(OUTPUT_DIR/'coverage_intervals.csv',dtype={'year':str})
paired=pd.read_csv(OUTPUT_DIR/'paired_differences.csv')
print(summary['status'])
print('Earlier selected methods:',selection['selected_methods'])
print('Daily update dates:',summary['daily_updates'])
policy=pd.DataFrame(selection['policy_scores'])
display(policy[['model','method','cases','flare_cases','nonflare_coverage','flare_coverage','max_class_coverage_deficit','mean_set_size','both_fraction']].round(4))''')
    md('## Results\n### 8. Fixed and selected methods\nThese are matched known-outcome windows. Review coverage together with ambiguity and support-guard frequency; including both labels can cover either outcome without making a decisive forecast.')
    code('''for alpha in CONFIG['alphas']:
    display(Markdown(f'**Target coverage: {1-alpha:.0%}**'))
    chosen=metrics[metrics.alpha.eq(alpha)&(metrics.method.eq('fixed_2015')|metrics.selected_on_policy)].copy()
    chosen['Period']=chosen.role.map(PERIOD_NAMES)
    display(chosen[['Period','model','method','cases','flare_cases','nonflare_coverage','flare_coverage','mean_set_size','both_fraction','support_guard_fraction']].round(4))
display(Markdown('**All declared candidate comparisons at the primary 90% target:**'))
display(metrics[metrics.alpha.eq(.1)][['role','model','method','selected_on_policy','nonflare_coverage','flare_coverage','both_fraction','support_guard_fraction']].round(4))''')
    md('### 9. Annual class coverage\nDots and lines show empirical coverage; shading shows conditional 95% region-component bootstrap intervals. The dotted line is the 90% prediction-set coverage target. It is distinct from the confidence level of the interval. The final year is partial.')
    code('''fig,axes=plt.subplots(2,2,figsize=(12,7.6),sharex=True,sharey=True,layout='constrained')
for row,model in enumerate(CONFIG['models']):
    chosen=selection['selected_methods'][model]
    methods=list(dict.fromkeys(['fixed_2015',chosen]))
    for col,population_name in enumerate(['nonflare','flare']):
        ax=axes[row,col]
        for method in methods:
            data=intervals[intervals.model.eq(model)&intervals.method.eq(method)&intervals.alpha.eq(.1)&intervals.population.eq(population_name)&intervals.year.ne('all')&intervals.grouping.eq('region_component')].copy()
            data['year']=data.year.astype(int);data=data.sort_values('year')
            fixed=method=='fixed_2015';color=COLORS['fixed_2015' if fixed else 'selected']
            ax.plot(data.year,data.coverage,marker='o' if fixed else 's',ls='-' if fixed else '--',color=color,label=method.replace('_',' '))
            ax.fill_between(data.year,data.low,data.high,color=color,alpha=.12)
        ax.axhline(.9,color='#444444',ls=':',lw=1.2)
        ax.set(title=f'{model.upper()} · {"No M/X" if population_name=="nonflare" else "M/X flare"}',ylim=(0,1.04),xticks=range(2021,2027),xticklabels=['2021','2022','2023','2024','2025','2026*'])
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.18)
        if col==0:ax.set_ylabel('Empirical set coverage');ax.legend(loc='lower left',frameon=False)
fig.suptitle('Fixed versus earlier-selected updating · 90% target · *2026 partial',fontsize=14)
fig.savefig(OUTPUT_DIR/'annual_coverage.png',bbox_inches='tight');plt.show()''')
    md('### 10. The cost in ambiguous forecasts\nBars contain only known-outcome evaluation windows. Missing inputs are not counted as empty sets. Colors and patterns distinguish the four possible set types.')
    code('''fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,layout='constrained')
styles=[('nonflare_only','Only no M/X','#245A81',''),('flare_only','Only M/X','#B36B19','//'),('both','Both labels','#D8B95B','..'),('empty','Empty','#92979C','xx')]
for ax,model in zip(axes,CONFIG['models']):
    chosen=selection['selected_methods'][model]
    table=metrics[metrics.model.eq(model)&metrics.alpha.eq(.1)&(metrics.method.eq('fixed_2015')|metrics.method.eq(chosen))].copy()
    table['order']=table.role.map({'retrospective_cycle25':0,'supplementary_2026':2})+table.method.ne('fixed_2015').astype(int)
    table=table.sort_values('order');left=np.zeros(len(table))
    for state,label,color,hatch in styles:
        values=table[state+'_fraction'].to_numpy()
        ax.barh(range(len(table)),values,left=left,color=color,hatch=hatch,edgecolor='white',label=label)
        for i,width in enumerate(values):
            if width>=.07:ax.text(left[i]+width/2,i,f'{width:.0%}',ha='center',va='center',color='white' if state=='nonflare_only' else '#202020')
        left+=values
    ax.set(yticks=range(len(table)),yticklabels=[f'{PERIOD_NAMES[r.role]}\\n{r.method.replace("_"," ")}' for r in table.itertuples()],xlim=(0,1),title=model.upper(),xlabel='Share of known-outcome evaluation windows')
    ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1))
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
fig.suptitle('Prediction-set composition · 90% target',fontsize=15)
fig.savefig(OUTPUT_DIR/'set_composition.png',bbox_inches='tight');plt.show()''')
    md('### 11. Low-support guards and ambiguity over time\nMonthly fractions use known-outcome windows. A support guard automatically includes one or both classes; it is not a validated operational fallback. This chart helps identify coverage obtained during periods with insufficient recent calibration support.')
    code('''fig,axes=plt.subplots(2,1,figsize=(12,6.2),sharex=True,sharey=True,layout='constrained')
for ax,model in zip(axes,CONFIG['models']):
    chosen=selection['selected_methods'][model]
    data=monthly[monthly.model.eq(model)&monthly.method.eq(chosen)&monthly.alpha.eq(.1)].sort_values('month')
    dates=pd.to_datetime(data.month+'-01')
    ax.plot(dates,data.both_fraction,color='#245A81',label='Both labels',lw=1.8)
    ax.plot(dates,data.support_guard_fraction,color='#B36B19',ls='--',label='Any class support guard',lw=1.8)
    ax.set(title=f'{model.upper()} · {chosen.replace("_"," ")}',ylim=(0,1.04),ylabel='Share of evaluated windows')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper right',frameon=False)
axes[-1].xaxis.set_major_locator(mdates.YearLocator());axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlim(pd.Timestamp('2021-01-01'),pd.Timestamp('2026-09-01'))
fig.suptitle('Ambiguity and insufficient calibration support · 90% target',fontsize=15)
fig.savefig(OUTPUT_DIR/'support_and_ambiguity.png',bbox_inches='tight');plt.show()''')
    md('### 12. Paired differences and complete population\nIntervals below compare the earlier-selected method with the fixed method on identical windows. They are conditional resampling sensitivities, not evidence of independent events or unconditional guarantees for the adaptive procedure.')
    code('''if len(paired):
    display(paired[paired.alpha.eq(.1)&paired.grouping.eq('region_component')][['role','model','method','metric','difference','low','high']].round(4))
availability=pd.read_csv(OUTPUT_DIR/'availability.csv')
display(availability[availability.year.ge(2021)].reset_index(drop=True))
support=pd.read_csv(OUTPUT_DIR/'support_summary.csv')
display(support[support.model.eq('gru')&support.alpha.eq(.1)][['lookback_days','label','updates','guarded_updates','minimum_windows','median_windows','minimum_components']])
for name,expected in summary['output_sha256'].items():assert sha256(OUTPUT_DIR/name)==expected,name
for directory in [PARENT_DIR,FIXED_DIR]:
    parent=json.loads((directory/'summary.json').read_text())
    for name,expected in parent['output_sha256'].items():assert sha256(directory/name)==expected,name
print('All saved artifacts and frozen parents verified.')
print('Candidate population:',summary['population_rows'],'| Missing inputs:',summary['missing_input_rows'])''')
    md('## Takeaways')
    code('''messages=[]
for model in CONFIG['models']:
    chosen=selection['selected_methods'][model]
    messages.append(f'**{model.upper()}:** earlier development selected **{chosen.replace("_"," ")}**.')
    for role in CONFIG['evaluation_roles']:
        rows=metrics[metrics.model.eq(model)&metrics.role.eq(role)&metrics.alpha.eq(.1)].set_index('method')
        fixed,selected=rows.loc['fixed_2015'],rows.loc[chosen]
        messages.append(f'{PERIOD_NAMES[role]}: flare coverage {fixed.flare_coverage:.1%} fixed → {selected.flare_coverage:.1%} selected; '
                        f'both-label sets {fixed.both_fraction:.1%} → {selected.both_fraction:.1%}; support guard on {selected.support_guard_fraction:.1%} of evaluated cases.')
display(Markdown('\\n\\n'.join(messages)))
display(Markdown('### Interpretation limits\\n'+'\\n'.join('- '+s for s in summary['limitations'])))''')
    md('''The next policy experiment must balance class coverage, incorrect decisive forecasts and forecast availability, including missing inputs and low-support periods. A singleton alone does not establish a trustworthy forecast. The earlier development block has now informed rolling-method selection; further tuning and validation must report that reuse transparently. AIA fusion and genuinely prospective logged forecasts remain later work.

**Reusable artifacts:** daily threshold journal with support and availability checks, full-population set codes, frozen earlier selection, all candidate metrics, monthly/yearly results, paired intervals and a source/runtime/hash contract. Per-case outputs remain local; compact receipts and the executed notebook are versioned.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (SHARP rolling)','language':'python'},'language_info':{'name':'python'},'colab':{'name':'04_SHARP_72h_Rolling_Conformal.ipynb','provenance':[]}})
    nbf.validate(nb);path=ROOT/'notebooks/04_SHARP_72h_Rolling_Conformal.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':build()
