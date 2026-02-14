import train_flowv2
import importlib
importlib.reload(train_flowv2)
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import warnings
from xgboost import XGBClassifier
from collections import defaultdict
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _compute_dist(y):
    """Distribution stats for y in {0,1}."""
    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "test_n": n,
        "test_pos_n": n_pos,
        "test_neg_n": n_neg,
        "test_pos_frac": (n_pos / n) if n else np.nan,
        "test_neg_frac": (n_neg / n) if n else np.nan,
    }

def walkback_runs(
    df,
    model,
    model_name,
    feature_cols,
    target_col,
    pi_year,
    *,
    date_col="Date",
    train_years=6,
    test_days=5,
    step_days=5,
    runs=20,
    purge_days=None,       # defaults to horizon_days
    fill_inf=0.0,
    min_feat=8,
    pi_handling=None,
    feature_dict=None,
    groups=None
):
    """
    Deployment-aligned evaluation:
      - For each run, take a 5-day OOT test window stepping back by 5 days.
      - Train on the prior N years (fixed-length window) ending right before test.
      - Purge 'purge_days' from the end of train to avoid overlap leakage for forward-return labels.
      - Score ONLY on the OOT test window (distribution + metrics).
    Returns: long DataFrame with one row per (feature_set/run/model).
    """
    rows = []
    pi_acc = defaultdict(lambda: {"pi_sum": 0.0, "count": 0})

    for k in range(runs):

        dfw = df.sort_values("Date").reset_index(drop=True).copy()

        n = len(dfw)
        train_size = 245 * int(train_years)
        test_size = int(test_days)
        step = int(step_days)
        purge = int(purge_days) if purge_days is not None else 0 #int(horizon_days)
        test_end = n - k * step
        test_start = test_end - test_size

        if test_start < 0:
            break

        train_end = test_start - purge
        train_start = train_end - train_size
        if train_start < 0 or train_end <= train_start:
            break
        
        dates = dfw[date_col].to_numpy() if date_col in dfw.columns else None
        dfpi = dfw[train_start:train_end].copy()

        if pi_handling == 'run_separately':

            final_features = []
            all_perm_dfs = []
            
            for g in groups:

                group_cols = feature_dict[g]
                perm_cols, p_df = train_flowv2.perm_list(
                    df=dfpi,
                    feature_cols=group_cols,
                    target_col=target_col,
                    model=model,
                    fill_inf=0.0,
                    pi_year=pi_year,
                    min_feats=min_feat,
                    feat_type=g
                )

                final_features += perm_cols
                all_perm_dfs.append(p_df)
                
                print(f"{g}: {len(group_cols)} | {len(perm_cols)} | {sorted(perm_cols)}")

            final_features = list(dict.fromkeys(final_features))

        elif pi_handling == 'include_new':

            perm_cols, perm_df = train_flowv2.perm_list(
                df=dfpi,
                feature_cols=feature_cols,
                target_col=target_col,
                model=model,
                fill_inf=0.0,
                pi_year=pi_year,
                min_feats=min_feat
            )

            print(f"{len(feature_cols)} | {len(perm_cols)} | All Cols: {sorted(perm_cols)}")
        
        # Drop any accidental return cols from features (belt+suspenders)
        safe_feature_cols = [c for c in final_features if not c.startswith("Return")]

        # Basic numeric cleaning
        dfw[safe_feature_cols] = dfw[safe_feature_cols].replace([np.inf, -np.inf], fill_inf)

        X_all = dfw[safe_feature_cols].to_numpy()
        #y_all = _to_binary(dfw[target_col].to_numpy())
        y_all = dfw[target_col].to_numpy()

        print(
            f"Run {k+1}/{runs} | "
            f"Train: {dates[train_start]} → {dates[train_end-1]} | "
            f"Test: {dates[test_start]} → {dates[test_end-1]} | "
            f"Train_n={train_end-train_start} | Test_n={test_end-test_start} | "
            f"(PI Years: {pi_year} - Feats: {min_feat})"
        )

        X_train = X_all[train_start:train_end]
        y_train = y_all[train_start:train_end]
        X_test  = X_all[test_start:test_end]
        y_test  = y_all[test_start:test_end]

        dist = _compute_dist(y_test)

        #start_time = time.time()
        m = clone(model)
        m.fit(X_train, y_train)

        preds = m.predict(X_test)
        proba = np.nan
        if hasattr(m, "predict_proba"):
            proba = float(m.predict_proba(X_test)[0, 1])   # prob(class=1)
        elif hasattr(m, "decision_function"):
            s = float(m.decision_function(X_test)[0])
            proba = float(1.0 / (1.0 + np.exp(-s)))        # squash to (0,1)
        proba = np.nan if np.isnan(proba) else round(round(proba / 0.05) * 0.05, 2)

        rows.append({
            "run": k + 1,
            "model": model_name,
            "test_days": test_days,
            "pred": round(proba,2),
            "acc": float(accuracy_score(y_test, preds)),
            **dist,
            "train_n": int(len(y_train)),
            "train_start": dates[train_start] if dates is not None else train_start,
            "train_end": dates[train_end - 1] if dates is not None else train_end - 1,
            "test_start": dates[test_start] if dates is not None else test_start,
            "test_end": dates[test_end - 1] if dates is not None else test_end - 1,
            "train_years": train_years,
            "n_features": len(safe_feature_cols),
            "pi_size": pi_year,
            "pi_handling": pi_handling,
            "min_feats": min_feat
        })

    return pd.DataFrame(rows)

def run_deploy_flow(days_assessed, r, pi_handling, feature_cols, df, model_name, model,
                   train_year, pi_year, min_feat, list_name, feature_dict, groups):

    results= []
    results_df = pd.DataFrame()

    runs = days_assessed

    base_cols = feature_cols
    train_year = train_year
    pi_year = pi_year
    min_feat = min_feat

    target_col = f"Return_{r}"
    # Trime unknown (recent) outcomes
    df_final = df.iloc[r:].copy()

    print(f"Running for horizon {r} | {pi_handling}")

    df_scores = walkback_runs(
        df=df_final,
        model=model,
        model_name=model_name,
        feature_cols=base_cols,
        target_col=target_col,
        pi_year=pi_year,
        date_col="Date",
        train_years=train_year,
        test_days=1,
        step_days=1,
        runs=runs,
        purge_days=r, 
        fill_inf=0.0,
        min_feat=min_feat,
        pi_handling=pi_handling,
        feature_dict=feature_dict,
        groups=groups
    )

    df_scores["features"] = list_name
    df_scores["horizon"] = r

    results.append(df_scores)

    results_df = pd.concat(results, ignore_index=True)

    return results_df