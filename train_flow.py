import min_features, daily_return
import importlib
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

importlib.reload(min_features)
importlib.reload(daily_return)

def import_data(ticker, minute_feats, returns):

    if minute_feats != 'N':

        df_min = min_features.min_features()
        df_daily, feature_sets = daily_return.pull_daily(ticker, returns) 

        df_main = pd.merge(df_min, df_daily, how='inner', on='Date')
        df_main = df_main.sort_values(by='Date', ascending=False)

        return_cols = df_main.columns[df_main.columns.str.contains("Return_")].to_list()
        daily_cols = [
            c for c in df_daily.iloc[:, 1:].columns
            if "return" not in c.lower()
        ]
        close_cols = df_min.columns[(df_min.columns.str.contains("close_")) | (df_min.columns.str.contains("post_")) | (df_min.columns.str.contains("overnight_"))].to_list()
        min_cols = (
            df_min
            .loc[:, ~df_min.columns.isin(close_cols)]  # drop close_ columns
            .iloc[:, 1:]                               # drop first column
            .columns
            .to_list()
        )
    else:
        df_daily, feature_sets = daily_return.pull_daily('QQQ', returns) 
        return_cols = df_daily.columns[df_daily.columns.str.contains("Return_")].to_list()
        daily_cols = [
            c for c in df_daily.iloc[:, 1:].columns
            if "return" not in c.lower()
        ]

    print(f'Available Feature Sets: {feature_sets.keys()}')

    ma_all_cols = feature_sets['ma']
    ma_lag = [c for c in ma_all_cols if "lag" in c.lower()]
    ma_rel = [c for c in ma_all_cols if "rel_" in c.lower()]
    ma_sma = [c for c in ma_all_cols if ("sma_" in c.lower()) and ("lag" not in c.lower())]
    ma_num = [c for c in ma_all_cols if ("num" in c.lower()) or ("since" in c.lower())]
    rsi_cols = feature_sets['rsi']
    macd_cols = feature_sets['macd']
    volu_cols = feature_sets['volume']
    atr_adx_cols = feature_sets['atr_adx']
    vola_cols = feature_sets['volatility']
    vix_skew_cols = feature_sets['vix_skew']
    experimental_slope_cols = feature_sets['experimental_slope']

    sets = [ma_lag, ma_rel, ma_sma, ma_num, rsi_cols + macd_cols, volu_cols, atr_adx_cols + vola_cols, vix_skew_cols, experimental_slope_cols]
    set_names = ["ma_lag", "ma_rel", "ma_sma", "ma_num", "rsi_macd", "volu", "atr_adx" + "vola", "vix_skew", "experimental_slope"]
    feature_sets = dict(zip(set_names,sets))
    feature_master_list = [x for sub in sets for x in sub]

    return df_daily, feature_sets, return_cols, daily_cols, feature_sets, feature_master_list

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
    models,
    feature_cols,
    target_col,
    pi_years,
    *,
    date_col="Date",
    train_years=6,
    test_days=5,
    step_days=5,
    runs=20,
    purge_days=None,       # defaults to horizon_days
    fill_inf=0.0,
    min_feats=8,
    pi_handling=None,
    new_features=None,
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
        new_features = new_features or []
        base_feature_cols = list(feature_cols)

        for model_name, model in models.items():

            for pi_year in pi_years:

                for min_feat in min_feats:

                    if pi_handling == 'exclude_new':

                        feat = [c for c in base_feature_cols if c not in new_features]

                        perm_cols, p_df = perm_list(
                            df=dfpi,
                            feature_cols=feat,
                            target_col=target_col,
                            model=model,
                            fill_inf=0.0,
                            pi_year=pi_year,
                            min_feats=min_feat
                        )

                        perm_cols += new_features
                        perm_cols = list(dict.fromkeys(perm_cols + new_features))
                        print(f"{len(feature_cols)} | {len(perm_cols)} | {sorted(perm_cols)}")
                        pi_value = f"{str(pi_year)}-{pi_handling[0]}"

                    elif pi_handling == 'run_separately':
                        
                        feat = [c for c in base_feature_cols if c not in new_features]

                        perm_cols, p_df = perm_list(
                            df=dfpi,
                            feature_cols=feat,
                            target_col=target_col,
                            model=model,
                            fill_inf=0.0,
                            pi_year=pi_year,
                            min_feats=min_feat
                        )

                        new_perm_cols, p_df = perm_list(
                            df=dfpi,
                            feature_cols=new_features,
                            target_col=target_col,
                            model=model,
                            fill_inf=0.0,
                            pi_year=pi_year,
                            min_feats=min_feat,
                            feat_type="New"
                        )
                        
                        print(f"{len(feature_cols)} | {len(perm_cols)} | Original Cols: {sorted(perm_cols)}")
                        print(f"{len(feature_cols)} | {len(new_perm_cols)} | New Cols: {sorted(new_perm_cols)}")
                        perm_cols += new_perm_cols
                        perm_cols = list(dict.fromkeys(perm_cols + new_features))
                        pi_value = f"{str(pi_year)}-{pi_handling[0]}"

                    elif pi_handling == 'include_new':

                        perm_cols, p_df = perm_list(
                            df=dfpi,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            model=model,
                            fill_inf=0.0,
                            pi_year=pi_year,
                            min_feats=min_feat
                        )
                        
                        pi_value = f"{str(pi_year)}-{pi_handling[0]}"

                        print(f"{len(feature_cols)} | {len(perm_cols)} | All Cols: {sorted(perm_cols)}")
                    
                    # Drop any accidental return cols from features (belt+suspenders)
                    safe_feature_cols = [c for c in perm_cols if not c.startswith("Return")]

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
                        "pi_size": pi_value,
                        "min_feats": min_feat
                    })

                    key_base = (model_name, len(safe_feature_cols), pi_value, min_feat, train_years)

                    for col, pi in zip(p_df["feature"], p_df["pi_mean"]):
                        key = key_base + (col,)
                        pi_acc[key]["pi_sum"] += float(pi)
                        pi_acc[key]["count"] += 1

    pi_rollup = (
        pd.DataFrame([
            {
                "model": key[0],
                "n_features": key[1],
                "pi_size": key[2],
                "min_feats": key[3],
                "train_years": key[4],
                "col_name": key[5],
                "pi_sum": v["pi_sum"],
                "count": v["count"],
                "pi_avg": v["pi_sum"] / v["count"],
            }
            for key, v in pi_acc.items()
        ])
        .sort_values(["model","train_years","n_features","pi_size","min_feats","pi_sum"],
                    ascending=[True,True,True,True,True,False])
        .reset_index(drop=True)
    )
        
    return pd.DataFrame(rows), pi_rollup

def perm_list(
    df,
    feature_cols,
    target_col,
    model,
    *,
    fill_inf=0.0,
    pi_year=1,
    min_feats=6,
    feat_type=None
):

    dfw = df.sort_values("Date").reset_index(drop=True).copy()
    
    # Drop any accidental return cols from features (belt+suspenders)
    safe_feature_cols = [c for c in feature_cols if not (c.startswith("Return"))]

    # Basic numeric cleaning
    dfw[safe_feature_cols] = dfw[safe_feature_cols].replace([np.inf, -np.inf], fill_inf)

    X_train = dfw[safe_feature_cols].to_numpy()
    y_train = dfw[target_col].to_numpy()
    #dates = dfw[date_col].to_numpy() if date_col in dfw.columns else None
    
    #N_PI = int(len(X_train) * perc_train)
    N_PI = int(242 * pi_year)
    #dates_pi = dates[-N_PI:]
    #print(f"PI Train: {min(dates_pi)} → {max(dates_pi)}")
    X_pi = X_train[-N_PI:]
    y_pi = y_train[-N_PI:]

    # fit model
    m = clone(model).fit(X_train, y_train)

    # permutation importance on training-only slice
    pi = permutation_importance(
        m,
        X_pi,
        y_pi,
        scoring="balanced_accuracy",   # or "accuracy", "neg_log_loss", etc.
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    # pi.importances_mean aligns to feature_cols order
    pi_df = pd.DataFrame({
        "feature": feature_cols,                 # same order used to build X_train
        "pi_mean": pi.importances_mean,
        "pi_std":  pi.importances_std,
    }).sort_values("pi_mean", ascending=False)
    #print(pi_df.head(8))

    if feat_type != "New":

        # keep only features with PI > 0
        pi_cols = pi_df['feature'][pi_df['pi_mean'] > .03].to_list()
        perm_df = pi_df[['feature', 'pi_mean']][pi_df['pi_mean'] > 0.03]
            
        if len(pi_cols) < min_feats:
            pi_cols = (
                pi_df.sort_values("pi_mean", ascending=False)
                    .head(min_feats)["feature"]
                    .tolist()
            )
        
        perm_df = pi_df[['feature', 'pi_mean']].sort_values("pi_mean", ascending=False).head(min_feats)
    
    else:

        # keep only features with PI > 0
        pi_cols = pi_df['feature'][pi_df['pi_mean'] > .03].to_list()
        perm_df = pi_df[['feature', 'pi_mean']][pi_df['pi_mean'] > 0.03]

        if len(pi_cols) < min_feats:
            pi_cols = (
                pi_df.sort_values("pi_mean", ascending=False)
                    .head(min_feats)["feature"]
                    .tolist()
            )

            perm_df = pi_df[['feature', 'pi_mean']].sort_values("pi_mean", ascending=False).head(min_feats)

    #print(pi_df.sort_values("pi_mean", ascending=False))
    #print(f"Ran permutation importance for horizon {purge_days} | Len: {N_PI} | Old: {len(feature_cols)} | New: {len(pi_cols)}")
    
    return pi_cols, perm_df

def run_train_flow(test_day, days_assessed, returns, pi_handlings, feature_cols, df, models,
                   train_years, pi_years, min_feats, list_name, new_features):

    results= []
    perm_results= []
    results_df = pd.DataFrame()
    perm_df = pd.DataFrame()
    runs = int(days_assessed / test_day)

    for pi_handling in pi_handlings:

        for r in returns:

            base_cols = feature_cols
            train_years = train_years
            pi_years = pi_years
            min_feats = min_feats

            target_col = f"Return_{r}"
            # Trime unknown (recent) outcomes
            df_final = df.iloc[r:].copy()

            for train_year in train_years:

                print(f"Running for horizon {r} | {pi_handling}")
                base_cols = list(dict.fromkeys(base_cols + new_features))

                df_scores, perm_df = walkback_runs(
                    df=df_final,
                    models=models,
                    feature_cols=base_cols,
                    target_col=target_col,
                    pi_years=pi_years,
                    date_col="Date",
                    train_years=train_year,
                    test_days=test_day,
                    step_days=test_day,
                    runs=runs,
                    purge_days=r, 
                    fill_inf=0.0,
                    min_feats=min_feats,
                    pi_handling=pi_handling,
                    new_features=new_features,
                )

                df_scores["feature_set"] = list_name
                df_scores["horizon"] = r
                perm_df["feature_set"] = f"kitch_sink_ba"
                perm_df["horizon"] = r

                results.append(df_scores)
                perm_results.append(perm_df)

    results_df = pd.concat(results, ignore_index=True)
    perm_df = pd.concat(perm_results, ignore_index=True)

    return results_df, perm_df.sort_values(by=["horizon", "feature_set", "pi_sum"], ascending=(True, True, False))

def pi_handling_test(pi_handlings):

    for pi_handling in pi_handlings:
        if pi_handling not in {"exclude_new", "include_new", "run_separately"}:
            raise ValueError(
                "Unknown permutation handling of new and hold features. "
                "Expected one of: exclude_new, include_new, run_separately."
            )