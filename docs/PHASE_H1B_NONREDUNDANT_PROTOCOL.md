# Phase H1b — nonredundant physical-state sensitivity protocol

Status: **PRE-EXECUTION STRUCTURAL CORRECTION — 6 October 2026**

H1b corrects a deterministic redundancy discovered after executing H1. It does not tune against Cycle-25 or 2026 outcomes and does not alter the frozen A–G policy.

## Reason for H1b

H1 represented each physical family by latest state, endpoint net change, and linear slope over three equally spaced 96-minute history steps. For points at -288, -192 and -96 minutes, the ordinary least-squares slope is exactly half the endpoint net change. Those two columns therefore encode the same information.

The correction is defined from algebra, not from observed H1 performance.

## Frozen nonredundant representation

For each predeclared physical family:

1. **latest** = state at issue minus 96 minutes;
2. **net_change** = state(issue-96) - state(issue-288);
3. **curvature** = state(issue-288) - 2*state(issue-192) + state(issue-96).

The second difference captures short-horizon nonlinearity/acceleration and is not determined by the endpoint net change.

All feature families, raw transforms, training support, input hashes and evidence boundaries remain identical to H1.

## Primary H1b questions

- Does the 2026 rise in physical Mahalanobis distance and q99 OOD rate survive removal of the redundant slope dimensions?
- Which physical-family levels/evolution terms shift most strongly?
- Does the interpretable physical model retain the same broad cross-regime degradation?
- Does the negative association between physical distance and case-level squared error persist?
- Are the results robust at active-region level?

No routing rule is selected from H1b. H2 is permitted only after H1b is reviewed.
