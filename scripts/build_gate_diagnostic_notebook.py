"""Build the self-contained notebook 06 diagnostic."""
import ast
import json
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]


def build():
    def functions(path):
        text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    shared=functions(ROOT/'scripts/decision_policy_sharp.py');diagnostic=functions(ROOT/'scripts/diagnose_sharp_gates.py')
    source=(ROOT/'scripts/decision_policy_sharp.py').read_text()
    rates=next(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RATE_DEFINITIONS' for t in n.targets))
    config=json.loads((ROOT/'configs/sharp72_gate_diagnostics_v1.json').read_text())
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# Which forecast checks remove useful flare alerts?
**Notebook 06 · Bamidele Akinwumi · diagnostic of frozen SHARP decisions**

## tl;dr
This notebook traces the decisions withheld by Notebook 05's four guards and decomposes the fallback's errors. All thresholds and predictions remain frozen. Tables and plots report exact window counts, overlapping guard triggers and paired conditional uncertainty; this is a diagnostic rather than selection of a new operational policy.''')
    md('''## Context & Methods

Notebook 05 found that guards reduce error among issued GRU forecasts while deferring many flare windows. Its logistic fallback restores some alerts but adds many incorrect decisions. This notebook asks **which checks account for that tradeoff, what changes if one check is removed, and what kinds of mistakes the fallback makes**.

Use the exact original GRU singleton decisions as the baseline. The four guards are insufficient calibration **support**, unusual feature **distance**, three-seed **spread**, and conflicting model singletons (**conflict**). Evaluate all 16 subsets at both parent conformal levels, without altering a threshold or refitting a model. Reproduce Notebook 05's all-guard decisions before analysing anything else.

### Key Assumptions

- **Standalone counts overlap:** a decision can trigger multiple guards. **Unique counts** show what dropping one guard recovers while retaining the other three. For a reconciled descriptive allocation, divide each removed case equally among its triggered guards. Fractional allocated counts are bookkeeping, not independent events or causal importance.
- Separate useful **true alerts** suppressed, **false alerts** prevented, incorrect **no-flare decisions** prevented and correct **no-flare decisions** suppressed. One prevented mistake need not have the same operational value as one lost flare alert.
- Paired group bootstraps compare a single guard removal against all guards on the same windows. Report candidate-minus-all-guards differences. They condition on realized decisions, do not rerun calibration or threshold fitting, and are not multiplicity-adjusted confirmatory tests.
- The 2015–2019 block is reused development evidence. The later years have already been inspected. No new winner is selected from them. The support-guard ablation is an accounting comparison, not approval to forecast with insufficient support.
- Candidate labels, window dependence, unverified historical delivery and source-quality limitations are inherited. These are forecast-window counts, not distinct flare-event detection rates. Unknown labels and missing inputs remain in the per-case output but never become fabricated negatives.
- The fallback is evaluated only on routed cases, which are a selected difficult subset. A high error there does not imply the same error over the logistic model's full population.

The diagnostic uses the already reviewed sources from Notebooks 04–05 and simple exact decomposition. It does not add an OOD detector, introduce a new uncertainty theorem or turn retrospective findings into prospective validation.''')
    md('## Data\n### 1. Pin the existing outputs\nUse the current training environment. No new downloads, training or cloud jobs are required. Functions are embedded and frozen files are checked before use.')
    imports='''from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd'''
    code(imports+'''
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Markdown
WORKSPACE=Path.cwd()
if WORKSPACE.name=='notebooks':WORKSPACE=WORKSPACE.parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':115,'axes.titleweight':'bold'})
PERIODS={'policy_validation':'2015–2019 development (reused)','retrospective_cycle25':'2021–2025','supplementary_2026':'2026 (partial)'}
GUARD_LABELS={'support':'Calibration support','distance':'Feature distance','spread':'Seed disagreement','conflict':'Model conflict'}
print({'numpy':np.__version__,'pandas':pd.__version__})''')
    code("CONFIG=json.loads(r'''"+json.dumps(config,indent=2)+"''')\n"+'''OUTPUT_DIR=WORKSPACE/'outputs'/('sharp72_gate_diagnostic_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
print('New run:',OUTPUT_DIR)
display(pd.DataFrame({'Guard':list(GUARD_LABELS.values()),'Bit':list(CONFIG['guard_bits'].values())}))''')
    md('### 2. Preserve metric definitions and frozen-source verification')
    code('\n\n'.join(shared[n] for n in ['sha256','save_json','checked_parent','decision_counts','decision_metrics'])+'\n\n'+rates)
    md('### 3. Reconcile all subsets and overlapping triggers\nThe same original label decision is either kept or withheld; removing a guard cannot change a 0 into a 1. Missing original decisions stay missing.')
    code('\n\n'.join(diagnostic[n] for n in ['validate_gate_inputs','subset_decisions','outcome_categories','gate_attribution','paired_guard_intervals']))
    md('### 4. Trace each case and calculate the diagnostic\nThe population output contains baseline decisions, guard bitmasks and fallback routes, sufficient to reconstruct each of the 16 comparisons.')
    code('\n\n'.join(diagnostic[n] for n in ['load_diagnostic_sources','build_case_diagnostic','analyze_cases','run_diagnostic']))
    md('### 5. Execute against the frozen runs')
    code('SOURCE_IMPORTS='+repr(imports)+'\n'+'''history=get_ipython().history_manager.input_hist_raw
starts=['def sha256','def validate_gate_inputs','def load_diagnostic_sources']
EXECUTED_SOURCE=SOURCE_IMPORTS+'\\n\\n'+'\\n\\n'.join(next(c for c in reversed(history) if c.startswith(s)) for s in starts)
summary=run_diagnostic(WORKSPACE,CONFIG,OUTPUT_DIR,EXECUTED_SOURCE)
metrics=pd.read_csv(OUTPUT_DIR/'subset_metrics.csv')
attribution=pd.read_csv(OUTPUT_DIR/'gate_attribution.csv')
overlaps=pd.read_csv(OUTPUT_DIR/'guard_overlap.csv')
intervals=pd.read_csv(OUTPUT_DIR/'paired_intervals.csv')
fallback=pd.read_csv(OUTPUT_DIR/'fallback_routes.csv')
yearly=pd.read_csv(OUTPUT_DIR/'yearly_attribution.csv')
print(summary['status'],'| subset comparisons:',summary['subset_comparisons'])
print('Full-population case/level rows:',summary['case_alpha_rows'])''')
    md('## Results\n### 6. Which guards account for useful alerts removed?\nAllocated counts share overlapping cases equally, so bars sum to the exact total removed. The table includes standalone and unique counts; these answer different questions.')
    code('''fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharex=True,layout='constrained')
for ax,role in zip(axes,['retrospective_cycle25','supplementary_2026']):
    table=attribution[attribution.role.eq(role)&attribution.alpha.eq(.1)].set_index('guard').loc[CONFIG['guards']]
    y=np.arange(4)
    for offset,column,label,color,hatch in [(-.2,'allocated_true_alert','Useful flare alerts suppressed','#B36B19','//'),(.2,'allocated_false_alert','False alerts prevented','#245A81','')]:
        vals=table[column].to_numpy();ax.barh(y+offset,vals,height=.36,color=color,hatch=hatch,label=label)
        for i,v in enumerate(vals):ax.annotate(f'{v:,.1f}',(v,i+offset),xytext=(4,0),textcoords='offset points',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=[GUARD_LABELS[g] for g in CONFIG['guards']],title=PERIODS[role],xlabel='Allocated forecast-window count');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
max_count=attribution[attribution.alpha.eq(.1)&attribution.role.ne('policy_validation')][['allocated_true_alert','allocated_false_alert']].to_numpy().max()
for ax in axes:ax.set_xlim(0,max_count*1.22)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('Guard attribution · fixed 90% parent target · shared overlaps',fontsize=14)
fig.savefig(OUTPUT_DIR/'guard_attribution.png',bbox_inches='tight');plt.show()
for role in ['policy_validation','retrospective_cycle25','supplementary_2026']:
    display(Markdown('**'+PERIODS[role]+'**'))
    display(attribution[attribution.role.eq(role)&attribution.alpha.eq(.1)][['guard','standalone_removed','unique_removed','allocated_true_alert','allocated_false_alert','allocated_false_clear','allocated_true_clear']].round(2).reset_index(drop=True))''')
    md('### 7. What does dropping one guard actually recover?\nCounts below are unique recoveries with the other three checks retained. They need not sum to the total because multiply-flagged cases remain withheld. Recovering an incorrect decision is a cost, not an improvement.')
    code('''for alpha in CONFIG['alphas']:
    display(Markdown(f'**Parent target {1-alpha:.0%}**'))
    table=attribution[attribution.alpha.eq(alpha)&attribution.role.ne('policy_validation')].copy();table['Period']=table.role.map(PERIODS)
    display(table[['Period','guard','unique_true_alert','unique_false_alert','unique_false_clear','unique_true_clear']].rename(columns={'unique_true_alert':'Flare alerts recovered','unique_false_alert':'False alerts restored','unique_false_clear':'False clears restored','unique_true_clear':'Correct clears recovered'}).reset_index(drop=True))
fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharex=True,layout='constrained')
for ax,role in zip(axes,['retrospective_cycle25','supplementary_2026']):
    table=intervals[intervals.role.eq(role)&intervals.alpha.eq(.1)&intervals.grouping.eq('region_component')]
    for offset,metric,label,color,marker in [(-.15,'alert_fraction_of_all_flares','Flare-alert fraction','#B36B19','o'),(.15,'false_clear_fraction_of_all_flares','False-clear fraction','#245A81','s')]:
        t=table[table.metric.eq(metric)].set_index('removed_guard').loc[CONFIG['guards']]
        ax.scatter(t.difference*100,np.arange(4)+offset,label=label,color=color,marker=marker)
        ax.hlines(np.arange(4)+offset,t.low*100,t.high*100,color=color,lw=1.8)
    ax.set(yticks=np.arange(4),yticklabels=[GUARD_LABELS[g] for g in CONFIG['guards']],xlabel='Change per 100 actual flare windows (percentage points)',title=PERIODS[role]);ax.axvline(0,color='#999',lw=.8);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('Remove one guard versus all guards · paired 95% region intervals',fontsize=14)
fig.savefig(OUTPUT_DIR/'single_guard_removal.png',bbox_inches='tight');plt.show()''')
    md('### 8. What makes the fallback error rate high?\nDecompose the selected fallback cases by why the GRU was withheld. These errors describe routed cases only; they do not establish the logistic model’s overall error. Each bar partitions all routed windows in that row.')
    code('''fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,layout='constrained')
route_order=['all_fallback','ambiguous_gru','spread_blocked_singleton'];route_labels=['All fallback cases','GRU has both labels','GRU singleton: high spread']
parts=[('true_alerts','Correct flare alert','#245A81',''),('false_alerts','False flare alert','#B05B4F','//'),('false_clears','Incorrect no-flare','#A47BB0','xx'),('true_clears','Correct no-flare','#B9C1C7','..')]
for ax,role in zip(axes,['retrospective_cycle25','supplementary_2026']):
    table=fallback[fallback.role.eq(role)&fallback.alpha.eq(.1)].set_index('route').loc[route_order];left=np.zeros(3)
    for field,label,color,hatch in parts:
        v=(table[field]/table.cases.replace(0,np.nan)).fillna(0).to_numpy();ax.barh(np.arange(3),v,left=left,color=color,hatch=hatch,edgecolor='white',label=label)
        for i,w in enumerate(v):
            if w>=.09:ax.text(left[i]+w/2,i,f'{w:.0%}',ha='center',va='center',color='white' if field in ['true_alerts','false_alerts'] else '#202020')
        left+=v
    ax.set(yticks=np.arange(3),yticklabels=[f'{label} (n={int(n):,})' for label,n in zip(route_labels,table.cases)],xlim=(0,1),xlabel='Share of routed known-outcome windows',title=PERIODS[role]);ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1))
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('Fallback outcomes by route · parent 90% target',fontsize=14)
fig.savefig(OUTPUT_DIR/'fallback_error_types.png',bbox_inches='tight');plt.show()
display(fallback[fallback.alpha.eq(.1)&fallback.role.ne('policy_validation')][['role','route','cases','true_alerts','false_alerts','false_clears','true_clears','alert_precision','clear_error_rate']].round(4).reset_index(drop=True))''')
    md('### 9. Retain the complete comparisons\nThe table shows primary-level later comparisons; the CSV preserves both levels and reused development results. This is diagnostic evidence, not a leaderboard for choosing a deployable winner.')
    code('''for role in ['retrospective_cycle25','supplementary_2026']:
    display(Markdown('**'+PERIODS[role]+'**'))
    display(metrics[metrics.role.eq(role)&metrics.alpha.eq(.1)][['active_guards','issued','selective_error','true_alerts','false_alerts','false_clears','deferred_flares']].round(4).reset_index(drop=True))
for name,digest in summary['output_sha256'].items():assert sha256(OUTPUT_DIR/name)==digest,name
print('Frozen artifacts verified; no operational policy changed.')''')
    md('## Takeaways')
    code('''messages=[]
for role in ['retrospective_cycle25','supplementary_2026']:
    t=attribution[attribution.role.eq(role)&attribution.alpha.eq(.1)].set_index('guard')
    largest=t.allocated_true_alert.idxmax();total=t.allocated_true_alert.sum();r=t.loc[largest]
    f=fallback[fallback.role.eq(role)&fallback.alpha.eq(.1)&fallback.route.eq('all_fallback')].iloc[0]
    messages.append(f'**{PERIODS[role]}:** {GUARD_LABELS[largest]} receives {r.allocated_true_alert:.1f} of {total:.0f} suppressed true-alert windows in the shared allocation. Dropping it alone would recover {int(r.unique_true_alert)} true alerts and restore {int(r.unique_false_alert)} false alerts plus {int(r.unique_false_clear)} false clears.')
    messages.append(f'The fallback makes {int(f.false_alerts)} false alerts and {int(f.false_clears)} false clears; its {int(f.alert_decisions)} positive decisions include {int(f.true_alerts)} correct flare alerts ({f.alert_precision:.1%} precision).')
display(Markdown('\\n\\n'.join(messages)))
display(Markdown('### Interpretation limits\\n'+'\\n'.join('- '+s for s in summary['limitations'])))''')
    md('''**Refinement to consider after this diagnostic:** treat flare alerts and no-flare decisions as separate operating decisions when defining error tolerance and escalation. Keep hard input/availability checks and the calibration-support requirement explicit. Investigate whether disagreement should trigger review rather than the current binary fallback, and evaluate that change in a declared future protocol. This notebook does not make that change or retune thresholds on 2026.

**Reusable outputs:** full-population guard masks and fallback routes, every guard subset, exact overlap counts, standalone/unique/shared contributions, annual contributions, paired conditional intervals and the frozen source contract.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (SHARP diagnostics)','language':'python'},'language_info':{'name':'python'},'colab':{'name':'06_SHARP_72h_Gate_Diagnostics.ipynb','provenance':[]}})
    nbf.validate(nb);p=ROOT/'notebooks/06_SHARP_72h_Gate_Diagnostics.ipynb';nbf.write(nb,p);print(p)

if __name__=='__main__':build()
