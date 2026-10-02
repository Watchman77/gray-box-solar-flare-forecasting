"""Build notebook 05 with tested decision-state functions embedded."""
import ast
import json
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]


def build():
    source=(ROOT/'scripts/decision_policy_sharp.py').read_text()
    tree=ast.parse(source)
    functions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    rates=next(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RATE_DEFINITIONS' for t in n.targets))
    imports='\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
    config=json.loads((ROOT/'configs/sharp72_decision_policy_v1.json').read_text())
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# 72-hour SHARP forecast-state decisions
**Notebook 05 · Bamidele Akinwumi · exploratory research prototype**

## tl;dr
This executed experiment evaluates which forecasts to issue and which to withhold. It reports errors among issued decisions alongside the share of all flare windows alerted, incorrectly cleared and deferred. The three states are experimental labels, not evidence of operational safety.''')
    md('''## Context & Methods

The uncertainty experiment improved flare inclusion by retaining both possible outcomes more often. Here we test a decision layer on the **same frozen GRU ensemble and logistic model**, using the previously selected **180-day rolling class-conditional sets**. There is no new backbone training or AIA download.

| Candidate state | Rule | Issued output |
|---|---|---|
| Normal | Valid assembled inputs; both calibration classes supported; GRU singleton; feature distance and seed spread within earlier-reference limits; no opposite logistic singleton | GRU probability and the singleton label |
| Degraded | Shared checks pass; GRU has both labels or excessive seed spread; logistic has its own supported singleton; no contradictory GRU singleton | Logistic probability and its singleton label; candidate fallback recorded |
| Abstain | Missing/invalid input, unavailable set, insufficient support, unusual feature distance, contradictory singletons, empty GRU set, or no eligible mode | No issued probability or binary decision; audit scores remain available |

The comparison includes ungated GRU singletons, support-gated GRU singletons, the full guards without fallback, and the full guards with fallback. These are declared ablations. A singleton's label determines the issued decision; this is not necessarily the same as thresholding its probability at 0.5.

### Key Assumptions

- The primary feature-distance limit is the **99th percentile** and seed-spread limit the **95th percentile**, measured on January–June 2014. Limits are heuristics fixed before this evaluation, not optimized on later labels or tied to an agreed acceptable-error target.
- Fit a Ledoit–Wolf covariance on flattened training histories from **2010–2013**, after the original signed-log standardization. Squared Mahalanobis distance measures deviation from that reference; it is not proof that a case is out of distribution. Three-seed probability standard deviation measures disagreement, not a calibrated uncertainty interval.
- The 2014 reference block already informed model early stopping. The 2015–2019 development block already selected rolling windows. Reuse is disclosed; none is fresh independent validation.
- Evaluate nominal conformal levels 90% and 95%. These are **parent set targets**, not accuracies promised for selected singletons. Rejecting cases does not preserve a conformal coverage guarantee.
- The same SHARP inputs support both models. The logistic fallback cannot repair missing SHARP, unusual shared features or source outages. It is a research candidate, not an approved reduced-input mode.
- Labels are provisional M/X-start-within-72-hour outcomes. Windows overlap. Unknown labels never count as negatives; missing inputs and unknown outcomes remain in full-population reporting.
- Historical delivery and source quality remain unverified; the parent replay assumes a 24-hour label delay after complete follow-up. Input-contract checks cover assembled finite histories and timestamp order, not all instrument quality conditions.
- All later years have already been inspected. This is exploratory retrospective work. Conditional group-bootstrap intervals do not include repeated gate fitting, sequential adaptation, selection or label uncertainty.

Selective error means **incorrect issued decisions / all issued decisions**. Retention means **issued decisions / evaluated windows**, distinct from conformal set coverage. Also report class-specific retention and the partition of all flare windows into alerts, false clears and deferrals. A low overall error can hide an unacceptable number of missed or deferred flares.

Primary methodological context: [Geifman & El-Yaniv, selective classification](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) motivates the risk–retention tradeoff; this notebook does not implement its risk certificate. [Ledoit–Wolf implementation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html) provides shrinkage covariance and squared distances.''')
    md('## Data\n### 1. Environment and configuration\nUse the existing training kernel and `requirements-notebook.txt`. All functions are embedded; no repository Python imports or cloud access are required. Exact parent files remain local and are verified by hashes.')
    code(imports+'''
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Markdown
WORKSPACE=Path.cwd()
if WORKSPACE.name=='notebooks':WORKSPACE=WORKSPACE.parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':115,'axes.titleweight':'bold'})
PERIODS={'policy_validation':'2015–2019 development (reused)','retrospective_cycle25':'2021–2025','supplementary_2026':'2026 (partial)'}
print({'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__})''')
    code("CONFIG=json.loads(r'''"+json.dumps(config,indent=2)+"''')\n"+'''OUTPUT_DIR=WORKSPACE/'outputs'/('sharp72_policy_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
print('New run:',OUTPUT_DIR)
display(pd.DataFrame(CONFIG['policies']))''')
    md('### 2. Verify sources and fit earlier reference checks\nOnly 2010–2013 training cases fit the covariance. The 2014 reference inputs set gate quantiles without reading their outcome values for threshold selection. Later cases do not change these frozen limits.')
    code('\n\n'.join(functions[n] for n in ['sha256','save_json','checked_parent','gate_threshold','load_policy_inputs']))
    md('### 3. Assign states without reading outcomes\nThe state function reads only scores, prediction sets, support and input checks. Its tests cover missing versus empty sets, conflicts, boundaries, invalid input, fallback restrictions and independence from current labels.')
    code(functions['decide_states'])
    md('### 4. Count errors and deferrals with explicit denominators\nUse 1,000 region-component resamples and 1,000 seven-day UTC block resamples as conditional sensitivities. They resample realized decisions, not the full adaptive training/calibration procedure.')
    code(rates+'\n\n'+'\n\n'.join(functions[n] for n in ['decision_counts','decision_metrics','decision_intervals','run_policy']))
    md('### 5. Execute and freeze outputs\nThe primary policy and sensitivities are recorded before evaluation. No later score selects a winner. Full per-case decisions, reasons and model references are saved locally.')
    code('SOURCE_IMPORTS='+repr(imports)+'\n'+'''history=get_ipython().history_manager.input_hist_raw
starts=['def sha256','def decide_states','RATE_DEFINITIONS']
EXECUTED_SOURCE=SOURCE_IMPORTS+'\\n\\n'+'\\n\\n'.join(next(c for c in reversed(history) if c.startswith(s)) for s in starts)
summary=run_policy(WORKSPACE,CONFIG,OUTPUT_DIR,EXECUTED_SOURCE)
gates=json.loads((OUTPUT_DIR/'frozen_gates.json').read_text())
metrics=pd.read_csv(OUTPUT_DIR/'metrics.csv')
states=pd.read_csv(OUTPUT_DIR/'state_metrics.csv')
intervals=pd.read_csv(OUTPUT_DIR/'intervals.csv')
availability=pd.read_csv(OUTPUT_DIR/'availability.csv')
reasons=pd.read_csv(OUTPUT_DIR/'reason_counts.csv')
yearly=pd.read_csv(OUTPUT_DIR/'yearly_metrics.csv')
print(summary['status'])
print('Covariance fit cases:',gates['fit_cases'],'| Reference cases:',gates['reference_cases'])
display(pd.DataFrame(gates['policies']).T.rename(columns={'distance':'Squared distance cutoff','spread':'Seed probability SD cutoff'}).round(5))''')
    md('## Results\n### 6. Errors, retention and flare outcomes\nThese tables use the same inherited known-outcome evaluation windows for every policy. The earlier block is development reuse. Compare selective error with classwise retention; more abstention is not automatically a scientific improvement.')
    code('''main=['gru_singleton','supported_gru','guarded_gru','guarded_fallback']
for alpha in CONFIG['alphas']:
    display(Markdown(f'**Parent conformal target: {1-alpha:.0%}**'))
    table=metrics[metrics.policy.isin(main)&metrics.alpha.eq(alpha)].copy()
    table['Period']=table.role.map(PERIODS)
    display(table[['Period','policy','cases','issued','selective_error','flare_retention','false_clear_fraction_of_all_flares','deferred_fraction_of_all_flares']].round(4).reset_index(drop=True))''')
    md('### 7. Error versus forecast retention\nEach point is a predeclared policy. The sensitivity points change both gate quantiles; they are descriptive, not tuned on these years. Labels identify the main ablations. Error bars are conditional 95% region-component bootstrap intervals for the principal comparisons.')
    code('''fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True,layout='constrained')
styles={'gru_singleton':('Ungated singleton','#8A5A44','s'),'supported_gru':('Support guard','#8576A6','D'),'guarded_gru':('Full guards, GRU','#245A81','^'),'guarded_fallback':('Full guards + fallback','#B36B19','o')}
for ax,role in zip(axes,['retrospective_cycle25','supplementary_2026']):
    table=metrics[metrics.role.eq(role)&metrics.alpha.eq(.1)]
    sens=table[~table.policy.isin(main)].sort_values('retention')
    ax.plot(sens.retention,sens.selective_error,color='#8D9398',ls=':',marker='.',label='Gate sensitivities')
    for name,(label,color,marker) in styles.items():
        row=table[table.policy.eq(name)].iloc[0]
        bounds=intervals[intervals.role.eq(role)&intervals.policy.eq(name)&intervals.alpha.eq(.1)&intervals.grouping.eq('region_component')&intervals.metric.eq('selective_error')]
        ax.scatter(row.retention,row.selective_error,color=color,marker=marker,s=70,label=label,zorder=3)
        if len(bounds) and bounds[['low','high']].notna().all(axis=None):
            lo,hi=bounds.iloc[0][['low','high']];ax.vlines(row.retention,lo,hi,color=color,lw=1.7)
    ax.set(title=PERIODS[role],xlabel='Retained share of known-outcome windows',xlim=(0,1),ylim=(0,max(.12,metrics.selective_error.max()*1.25)))
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.17)
axes[0].set_ylabel('Incorrect share of issued decisions')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Selective error and forecast availability · parent 90% target',fontsize=14)
fig.savefig(OUTPUT_DIR/'risk_retention.png',bbox_inches='tight');plt.show()''')
    md('### 8. What happens to all flare windows?\nThese bars sum to 100% of actual flare windows in each matched evaluation period. Deferrals are kept separate from incorrect no-flare decisions; neither is a successful flare alert.')
    code('''fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,layout='constrained')
parts=[('alert_fraction_of_all_flares','Flare alert','#245A81',''),('false_clear_fraction_of_all_flares','Incorrect no-flare decision','#B05B4F','//'),('deferred_fraction_of_all_flares','Deferred','#C7CDD2','..')]
for ax,role in zip(axes,['retrospective_cycle25','supplementary_2026']):
    table=metrics[metrics.role.eq(role)&metrics.alpha.eq(.1)].set_index('policy').loc[main];left=np.zeros(len(table))
    for field,label,color,hatch in parts:
        values=table[field].to_numpy();ax.barh(range(len(table)),values,left=left,color=color,hatch=hatch,edgecolor='white',label=label)
        for i,v in enumerate(values):
            if v>.07:ax.text(left[i]+v/2,i,f'{v:.0%}',ha='center',va='center',color='white' if field!='deferred_fraction_of_all_flares' else '#202020')
        left+=values
    ax.set(yticks=range(len(table)),yticklabels=[styles[n][0] for n in main],title=f'{PERIODS[role]} · {int(table.flare_cases.iloc[0]):,} flare windows',xlim=(0,1),xlabel='Share of all known flare windows');ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1))
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Flare alerts, false clears and deferrals · parent 90% target',fontsize=14)
fig.savefig(OUTPUT_DIR/'flare_outcomes.png',bbox_inches='tight');plt.show()''')
    md('### 9. Candidate states across the entire calendar population\nThese denominators include missing inputs, unknown outcomes and year-boundary cases. They differ from the performance tables. The state names are experimental; source-level quality and actual delivery remain unverified.')
    code('''fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
table=availability[availability.alpha.eq(.1)].sort_values('year');bottom=np.zeros(len(table))
for field,label,color,hatch in [('normal','Candidate normal','#245A81',''),('degraded','Candidate fallback','#B36B19','//'),('abstain','Abstain','#C7CDD2','..')]:
    values=table[field]/table.population_cases
    ax.bar(table.year.astype(str).replace('2026','2026*'),values,bottom=bottom,color=color,hatch=hatch,edgecolor='white',label=label);bottom+=values
ax.set(ylim=(0,1),ylabel='Share of all candidate windows',title='Full-population state availability · primary policy · *2026 partial')
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False)
fig.savefig(OUTPUT_DIR/'population_states.png',bbox_inches='tight');plt.show()
display(table[['year','population_cases','missing_inputs','normal','degraded','abstain','issued_unknown_outcomes','known_flares','deferred_known_flares']].reset_index(drop=True))''')
    md('### 10. Fallback errors and reasons for withholding forecasts\nThe fallback is evaluated separately. A low aggregate error can reflect a predominance of quiet windows; retain the flare counts and errors in view. Reason counts use precedence so each case has one main explanation; the full output also saves overlapping primary flags.')
    code('''table=states[states.policy.eq(CONFIG['primary_policy'])&states.alpha.eq(.1)&states.role.ne('policy_validation')].copy()
table['Period']=table.role.map(PERIODS)
display(table[['Period','state','cases','flare_cases','issued','errors','selective_error','true_alerts','false_clears','deferred_flares']].round(4).reset_index(drop=True))
display(reasons[reasons.alpha.eq(.1)&reasons.role.ne('policy_validation')][['role','reason','cases','flare_cases']].reset_index(drop=True))
display(intervals[intervals.policy.eq(CONFIG['primary_policy'])&intervals.alpha.eq(.1)&intervals.role.ne('policy_validation')&intervals.metric.isin(['selective_error','false_clear_fraction_of_all_flares','deferred_fraction_of_all_flares'])][['role','grouping','metric','low','high']].round(4).reset_index(drop=True))''')
    md('## Takeaways')
    code('''messages=[]
for role in ['retrospective_cycle25','supplementary_2026']:
    rows=metrics[metrics.role.eq(role)&metrics.alpha.eq(.1)].set_index('policy')
    base=rows.loc['gru_singleton'];chosen=rows.loc[CONFIG['primary_policy']]
    messages.append(f'**{PERIODS[role]}:** primary candidate retains **{chosen.retention:.1%}** of evaluated windows (ungated singletons: {base.retention:.1%}); errors among issued decisions are **{chosen.selective_error:.1%}** (ungated: {base.selective_error:.1%}).')
    messages.append(f'Of all flare windows, **{chosen.alert_fraction_of_all_flares:.1%}** receive a flare alert, **{chosen.false_clear_fraction_of_all_flares:.1%}** an incorrect no-flare decision, and **{chosen.deferred_fraction_of_all_flares:.1%}** are deferred.')
display(Markdown('\\n\\n'.join(messages)))
display(Markdown('### Limits on interpretation\\n'+'\\n'.join('- '+s for s in summary['limitations'])))
for name,expected in summary['output_sha256'].items():assert sha256(OUTPUT_DIR/name)==expected,name
print('Frozen artifacts verified; operationally validated: False')''')
    md('''**Next scientific decision:** assess whether the reduction in incorrect decisive forecasts justifies the loss of useful flare alerts and the extra fallback errors. Operational cost and escalation requirements should be agreed before declaring an acceptable policy. Further tuning reuses already inspected evidence and needs a fresh future evaluation for confirmation.

**Reusable outputs:** state and reason per case, frozen reference limits and distance estimator, all declared policy comparisons, classwise denominators, annual summaries, two conditional resampling sensitivities, source hashes and executed code. Full per-case files remain local; compact results and this executed notebook are versioned.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (SHARP policy)','language':'python'},'language_info':{'name':'python'},'colab':{'name':'05_SHARP_72h_Decision_Policy.ipynb','provenance':[]}})
    nbf.validate(nb);p=ROOT/'notebooks/05_SHARP_72h_Decision_Policy.ipynb';nbf.write(nb,p);print(p)

if __name__=='__main__':build()
