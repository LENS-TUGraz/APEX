"""
Predict reliability *and* energy for unseen RPL parameter sets.
Creates three separate records (IDs) per parameter set.

Output JSON shape
-----------------
{
  "id": <int>,
  "params": { "_RPL_DIO_INTERVAL_MIN": 8, ... },
  "metrics": {
    "reliability": <float>,          # synthetic sample
    "energy":      <float>,          # synthetic sample
    "var_reliability": <float>,      # σ² used to draw this group
    "var_energy":       <float>,
    "latency": NaN
  }
}
"""

import itertools, json, numpy as np, pandas as pd
from pathlib import Path
from statistics import mean
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------ #
# 0  File with the real-world data set                                                       #
# ------------------------------------------------------------------ #
RAW_FILE = Path("../RPL_results.json")


# ------------------------------------------------------------------ #
# 1  Load and flatten                                                #
# ------------------------------------------------------------------ #
df = pd.json_normalize(json.load(RAW_FILE.open()))
PARAM_COLS = [c for c in df.columns if c.startswith("params.")]
PRR_COL    = "metrics.reliability"
ENE_COL    = "metrics.energy"

# ------------------------------------------------------------------ #
# 2  Aggregate mean & variance                                       #
# ------------------------------------------------------------------ #
agg = (df.groupby(PARAM_COLS, dropna=False)
         .agg(mean_reliability=(PRR_COL, "mean"),
              var_reliability =(PRR_COL, "var"),
              mean_energy     =(ENE_COL, "mean"),
              var_energy      =(ENE_COL, "var"))
         .reset_index())

# ------------------------------------------------------------------ #
# 3  Random-Forest models on *variance*                              #
# ------------------------------------------------------------------ #
def build_variance_model(var_series):
    X = agg[PARAM_COLS]
    y = agg[var_series].fillna(0.0)
    return Pipeline([
        ("enc", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), PARAM_COLS)])),
        ("rf",  RandomForestRegressor(n_estimators=400, random_state=42))
    ]).fit(X, y)

rf_var_prr  = build_variance_model("var_reliability")
rf_var_ene  = build_variance_model("var_energy")

# ------------------------------------------------------------------ #
# 4  Design space                                                    #
# ------------------------------------------------------------------ #
SPACE = {
    "params._RPL_DIO_INTERVAL_MIN": [8, 12, 16],
    "params._MAX_LINK_METRIC":      [2048, 4096, 8192],
    "params._RANK_THRESHOLD":       [192, 384, 768],
    "params._RPL_DAO_DELAY":        [256, 512, 1024],
    "params._RPL_DIS_INTERVAL":     [1280, 3840, 7680],
    "params._RPL_PROBING_INTERVAL": [11520],  # Only this value allowed
}

KEYS   = list(SPACE.keys())
GRID   = [dict(zip(KEYS, combo))
          for combo in itertools.product(*SPACE.values())]
measured = set(agg[KEYS].apply(tuple, axis=1))
unseen   = [p for p in GRID if tuple(p[k] for k in KEYS) not in measured]

# ------------------------------------------------------------------ #
# 5  Rule-based mean PRR (unchanged)                                 #
# ------------------------------------------------------------------ #
BASE_PRR = df[PRR_COL].mean()
Δ_PRR = {
    "_RPL_DIO_INTERVAL_MIN": {4:-0.02,6:-0.01,8:0.02,12:0.01,14:-0.01,16:-0.02},
    "_MAX_LINK_METRIC":      {1024:0.01,2048:-0.17,4096:0.09,8192:0.085},
    "_RANK_THRESHOLD":       {192:-0.01,384:0.0,576:0.005,768:0.01},
    "_RPL_DAO_DELAY":        {256:-0.01,512:0.0,1024:0.005},
    "_RPL_DIS_INTERVAL":     {1280:0.01,3840:0.0,7680:-0.01},
    "_RPL_PROBING_INTERVAL": {5760:0.02,11520:0.0,23040:-0.02},
}
def mean_prr(p):
    x = BASE_PRR
    for short, delta in Δ_PRR.items():
        x += delta[p[f"params.{short}"]]
    return max(0.05, min(0.995, round(x, 6)))

# ------------------------------------------------------------------ #
# 6  Rule-based mean ENERGY (mJ)                                     #
# ------------------------------------------------------------------ #
BASE_ENE = df[ENE_COL].mean()          # ≈ 2 900 mJ on your dataset
Δ_ENE = {
    "_RPL_DIO_INTERVAL_MIN": {4:60,6:35,8:15,12:0,16:-10},
    "_MAX_LINK_METRIC":      {2048:0,4096:20,8192:35},
    "_RANK_THRESHOLD":       {192:10,384:0,768:8},
    "_RPL_DAO_DELAY":        {256:5,512:0,1024:-10},
    "_RPL_DIS_INTERVAL":     {1280:15,3840:0,7680:-8},
    "_RPL_PROBING_INTERVAL": {5760:25,11520:0,23040:-20},
}
def mean_energy(p):
    x = BASE_ENE
    for short, delta in Δ_ENE.items():
        x += delta[p[f"params.{short}"]]
    # plausible bounds
    return round(max(2500, min(3500, x)), 2)

# ------------------------------------------------------------------ #
# 7  Draw 3 samples, emit one record each                            #
# ------------------------------------------------------------------ #
rng = np.random.default_rng(42)
next_id = int(df["id"].max()) + 1 if "id" in df.columns else 1
records = []

for params in unseen:
    # Variances
    σ2_prr = max(float(rf_var_prr.predict(pd.DataFrame([params]))[0]), 1e-6)
    σ2_ene = max(float(rf_var_ene.predict(pd.DataFrame([params]))[0]), 1.0)

    μ_prr = mean_prr(params)
    μ_ene = mean_energy(params)

    prr_samples = np.clip(rng.normal(μ_prr, np.sqrt(σ2_prr), 4), 0.0, 0.995)
    ene_samples = np.clip(rng.normal(μ_ene, np.sqrt(σ2_ene), 4), 2500, 3500)

    for prr_s, ene_s in zip(prr_samples, ene_samples):
        records.append({
            "id": next_id,
            "params": {k.replace("params.",""): v for k, v in params.items()},
            "metrics": {
                "reliability": round(float(prr_s), 6),
                "energy":      round(float(ene_s), 2),
                "latency": np.nan
            }
        })
        next_id += 1

# ------------------------------------------------------------------ #
# 8  Save                                                            #
# ------------------------------------------------------------------ #
# old real testbed data filterd only for three values for each parameter
with open("filtered_results_3_3_3_4.json", "r") as f:
    real_data = json.load(f)

# Combine with real data
real_records_combined = real_data + records

length_dict_records = len(real_records_combined)
OUT_FILE = Path(f"real_synthetic_combined_{length_dict_records}_5.json")
json.dump(real_records_combined, OUT_FILE.open("w"), indent=2, allow_nan=True)
print(f"✓ Wrote {len(records)} records to {OUT_FILE.resolve()}")
