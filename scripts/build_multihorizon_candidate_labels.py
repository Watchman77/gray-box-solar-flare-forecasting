"""Build provisional 24 h and 3 h labels from the versioned event adjudication.

This is a label audit only. It does not assign experiment roles or authorize training.
"""
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
try:
    from scripts.build_dataset_inventory import tai_to_utc
except ModuleNotFoundError:
    from build_dataset_inventory import tai_to_utc

KNOWN_DECISIONS = {"original_science_not_in_review_queue", "retained_original_daily_supported", "filled_missing_direct_xra"}

def parse_regions(value):
    if not re.fullmatch(r"\d+(?:,\d+)*", str(value)):
        raise ValueError(f"Unresolved NOAA association: {value}")
    return tuple(sorted(set(int(x) for x in str(value).split(","))))

def event_index(events):
    out = {}
    for region, part in events.dropna(subset=["noaa_full_id_candidate"]).groupby("noaa_full_id_candidate"):
        part = part.sort_values(["start_time", "event_id"])
        out[int(region)] = (part.start_time.astype("int64").to_numpy(), part.event_id.astype(str).to_numpy())
    return out

def ids_in(index, start, end):
    if index is None: return ()
    t, ids = index
    a, b = np.searchsorted(t, [start, end], side="right")
    return tuple(ids[a:b])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=Path("data/processed/gray_box_inventory_v2/forecast_case_inventory.csv.gz"))
    ap.add_argument("--events", type=Path, default=Path("data/processed/gray_box_outcomes_v2/adjudicated_events.csv.gz"))
    ap.add_argument("--leap", type=Path, default=Path("data/raw/gray_box_inventory_v1/Leap_Second.dat"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/gray_box_multihorizon_labels_v1"))
    args = ap.parse_args()
    if args.output.exists(): raise ValueError("Output exists; choose a new version")
    cases = pd.read_csv(args.inventory, low_memory=False)
    events = pd.read_csv(args.events, low_memory=False)
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events["time"] = pd.to_datetime(events["time"], utc=True)
    events["is_mx"] = events["flare_class"].astype(str).str.startswith(("M", "X"))
    events = events.loc[events.is_mx].copy()
    known = events.region_decision.isin(KNOWN_DECISIONS)
    unknown_index = (events.loc[~known].start_time.astype("int64").to_numpy(), events.loc[~known].event_id.astype(str).to_numpy())
    by_region = event_index(events.loc[known])
    empty = (np.array([], dtype="int64"), np.array([], dtype=str))
    native = pd.to_datetime(cases.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI")
    args.output.mkdir(parents=True)
    all_rows=[]; support=[]
    for h in (24, 3):
        ends = tai_to_utc((native + pd.Timedelta(hours=h)).dt.strftime("%Y.%m.%d_%H:%M:%S_TAI"), args.leap)
        starts = pd.to_datetime(cases.issue_utc, utc=True).astype("int64").to_numpy()
        end_ns = pd.to_datetime(ends, utc=True).astype("int64").to_numpy()
        rows=[]
        for c, st, en in zip(cases.itertuples(), starts, end_ns):
            regions=parse_regions(c.NOAA_ARS)
            primary=ids_in(by_region.get(int(c.NOAA_AR_clean), empty), st, en)
            patch=tuple(sorted({e for r in regions for e in ids_in(by_region.get(r, empty), st, en)}))
            unknown=ids_in(unknown_index, st, en)
            # The event catalogue envelope is treated conservatively; no role is assigned here.
            within = st >= pd.Timestamp("2010-01-01", tz="UTC").value and en <= pd.Timestamp("2026-10-01", tz="UTC").value
            rows.append({"forecast_case_id":c.forecast_case_id,"horizon_hours":h,"issue_utc":c.issue_utc,"outcome_end_utc":ends.iloc[len(rows)],
                "primary_event_ids":";".join(primary),"patch_event_ids":";".join(patch),"unknown_mx_event_ids":";".join(unknown),
                "candidate_primary_label":(1 if primary else (pd.NA if unknown or not within else 0)),
                "candidate_patch_label":(1 if patch else (pd.NA if unknown or not within else 0)),
                "nominal_span_contains_window":within,"label_status":"provisional_not_training_truth","new_experiment_role":"unassigned","training_ready":False})
        frame=pd.DataFrame(rows); frame["candidate_primary_label"]=frame.candidate_primary_label.astype("Int64"); frame["candidate_patch_label"]=frame.candidate_patch_label.astype("Int64")
        frame.to_csv(args.output/f"candidate_labels_{h}h.csv.gz", index=False, compression="gzip")
        for scope in ("primary","patch"):
            lab=frame[f"candidate_{scope}_label"]
            support.append({"horizon_hours":h,"scope":scope,"cases":len(frame),"positive":int((lab==1).sum()),"negative":int((lab==0).sum()),"unresolved":int(lab.isna().sum())})
        all_rows.append(frame)
    pd.DataFrame(support).to_csv(args.output/"support.csv",index=False)
    receipt={"status":"candidate_labels_built_not_training_ready","inventory":str(args.inventory),"events":str(args.events),"horizons":[24,3],"known_event_decisions":sorted(KNOWN_DECISIONS),"support":support}
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
