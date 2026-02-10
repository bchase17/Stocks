import pandas as pd
import numpy as np
import min_features, daily_return
import importlib
importlib.reload(min_features)
importlib.reload(daily_return)
from sklearn.metrics import brier_score_loss, log_loss, matthews_corrcoef, balanced_accuracy_score

def import_data(file, df):

    perf_df = pd.read_csv(file)
    #perf_df = perf_df.drop_duplicates()
    perf_df['Date'] = perf_df['test_start']
    return_cols = df.columns[df.columns.str.contains("Return_")].to_list()

    return return_cols, perf_df

def _clip01(p, eps=1e-15):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)

def _metrics(y_col, p_col, min_th, cov_th, g: pd.DataFrame) -> pd.Series:

    y = g[y_col].astype(int).to_numpy()
    p = _clip01(g[p_col].to_numpy())

    yhat = np.where(p >= min_th, 1, np.where(p <= (1 - min_th), 0, np.nan))

    # keep only confident predictions (>=min_th in either direction)
    conf = ~np.isnan(yhat)
    y = y[conf]
    p = p[conf]
    yhat = yhat[conf].astype(int)
    records = int(len(y))

    # confident subset mask
    sel = (p >= cov_th) | (p <= (1-cov_th))

    # probability metrics
    brier = brier_score_loss(y, p)
    ll = log_loss(y, p)

    # MCC (needs both classes in y and yhat)
    mcc = np.nan
    if (np.unique(y).size > 1) and (np.unique(yhat).size > 1):
        mcc = matthews_corrcoef(y, yhat)

    # Balanced accuracy (needs both classes in y)
    bal_acc = np.nan
    pos_acc = np.nan
    neg_acc = np.nan

    if np.unique(y).size > 1:
        bal_acc = balanced_accuracy_score(y, yhat)

        # Positive class accuracy (TPR)
        pos_mask = (y == 1)
        if pos_mask.any():
            pos_acc = float((yhat[pos_mask] == 1).mean())

        # Negative class accuracy (TNR)
        neg_mask = (y == 0)
        if neg_mask.any():
            neg_acc = float((yhat[neg_mask] == 0).mean())

    # confident accuracy + coverage
    cov = float(sel.mean())
    acc_conf = float((yhat[sel] == y[sel]).mean()) if sel.any() else np.nan
    
    prec_pos_all = np.nan
    prec_neg_all = np.nan
    prec_pos_th = np.nan
    prec_neg_th = np.nan

    # positive-class precision
    pred_pos = (yhat == 1)
    if pred_pos.any():
        prec_pos_all = float((y[pred_pos] == 1).mean())   # TP/(TP+FP)

    pred_neg = (yhat == 0)
    if pred_neg.any():
        prec_neg_all = float((y[pred_neg] == 0).mean())   # TN/(TN+FN)

    # positive-class precision
    pred_pos = (yhat == 1) & sel
    if pred_pos.any():
        prec_pos_th = float((y[pred_pos] == 1).mean())

    # negative-class precision
    pred_neg = (yhat == 0) & sel
    if pred_neg.any():
        prec_neg_th = float((y[pred_neg] == 0).mean())

    return pd.Series({
        "pos_rate": float(y.mean()),
        "records": records,
        "bal_prec": round((prec_pos_all + prec_neg_all) / 2, 2),
        f"bal_prec_|{cov_th}|": round((prec_pos_th + prec_neg_th) / 2, 2),
        "brier": float(brier),
        "log_loss": float(ll),
        "mcc": float(mcc) if not np.isnan(mcc) else np.nan,
        "pprec": prec_pos_all, 
        "nprec": prec_neg_all,
        f"cov_|{cov_th}|": cov,
        f"pprec_|{cov_th}|": prec_pos_th,
        f"nprec_|{cov_th}|": prec_neg_th,
    })

def run_performance(perf_df, min_th, cov_th):

    gcols = ["horizon", "model", "train_years", "feature_set", "pi_size", "min_feats"]

    y_col = "test_pos_n"   # 0/1 actual
    p_col = "pred"         # P(y=1)

    pred_df = perf_df[(perf_df["pred"] > min_th) | (perf_df["pred"] < (1 - min_th))].copy()
    metrics_df = (
        pred_df
        .dropna(subset=[y_col, p_col])
        .groupby(gcols, sort=False)
        .apply(lambda g: _metrics(y_col, p_col, min_th, cov_th, g), include_groups=False)
        .reset_index()
        .sort_values(["horizon", "mcc", "brier"], ascending=[True, False, True])
    )

    metrics_df['composite'] =  0.5*(metrics_df['bal_prec']) + 0.25*(metrics_df['mcc']) + .125*(1-(metrics_df['brier'])) + 0.1*((metrics_df[f'bal_prec_|{cov_th}|']) * (metrics_df[f'cov_|{cov_th}|']))
    # top per horizon (ranked by MCC desc, then Brier asc)
    top_by_horizon = (
        metrics_df
        .sort_values(["horizon", "composite", "log_loss"], ascending=[True, False, False])
        .groupby("horizon", as_index=False, sort=False)
        .head(10)
    )

    return top_by_horizon.round(2)

def flip_bucket_tables_multi_dual(
    df_daily,
    perf_df,
    returns,
    min_th,
    *,
    K=3,
    date_col="Date",
    close_col="Close",
    w=None,
    perf_filter=None,  # optional callable to filter perf_df per horizon
):
    """
    For each horizon r:
      - builds streak + streak_lag1 from Return_r
      - merges into perf_df rows for (horizon=r) (and any extra filters you provide)
      - buckets BOTH streak and streak_lag1 into [-K..K] plus tails as +/- (K+1) labeled "3+"
      - computes:
          - wba_close (from streak) and wba_open (from streak_lag1)
          - bal_acc pair scores for +/-1, +/-2, +/-3 for both contexts
          - (optional) keeps acc/n wide columns for each context with suffixes _c and _o

    Returns:
      by_r: dict[r] -> wide table (flat columns)
      all_out: concat of all horizons with horizon as index level (flat columns)
    """
    if w is None:
        # weights: ±1 -> 2.0, ±2 -> 1.5, ±3 -> 1.25, ±3+ -> 1.0
        w = {1: 2.0, 2: 1.5, 3: 1.25, "3+": 1.0}

    max_score = float(sum(w.values()))
    gcols = ["model", "train_years", "feature_set", "pi_size", "min_feats"]

    def _add_streak(df_base, ret_col):
        d = df_base[[date_col, close_col, ret_col]].sort_values(date_col).copy()
        s = d[ret_col].astype("int8")
        grp = s.ne(s.shift()).cumsum()
        streak_len = s.groupby(grp).cumcount() + 1
        d["streak"] = streak_len.where(s.eq(1), -streak_len).astype("int32")
        d["streak_lag1"] = d["streak"].shift(1).fillna(0).astype("Int64")
        return d

    def _bucketize(series: pd.Series) -> pd.Series:
        b = series.clip(lower=-K, upper=K).astype("int32")
        b = b.copy()
        b.loc[series < -K] = -(K + 1)
        b.loc[series >  K] =  (K + 1)
        return b

    def _make_context_table(d: pd.DataFrame, bucket_col: str, suffix: str) -> pd.DataFrame:
        """
        suffix: "_c" for streak (close), "_o" for streak_lag1 (open)
        returns a flat-column wide df indexed by gcols, containing:
          - acc_{bucket}{suffix}, n_{bucket}{suffix} for buckets {1,-1,2,-2,3,-3,3+,-3+}
          - bal_acc_{1/2/3}{suffix}
          - wba_{close/open} (handled outside)
        """
        flip_perf = (
            d.groupby(gcols + [bucket_col], sort=False)
             .agg(n=("acc", "size"), acc=("acc", "mean"))
        )

        wide = pd.concat(
            {"acc": flip_perf["acc"].unstack(bucket_col),
             "n":   flip_perf["n"].unstack(bucket_col)},
            axis=1
        )

        # enforce order: +1/-1, +2/-2, +3/-3, 3+/-3+
        ordered_cols = []
        for k in [1, 2, 3, "3+"]:
            pb = (K + 1) if k == "3+" else k
            nb = -(K + 1) if k == "3+" else -k
            ordered_cols += [("acc", pb), ("acc", nb), ("n", pb), ("n", nb)]
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))

        # relabel bucket keys to strings ("3+", "-3+", etc.)
        rename_cols = []
        for metric, b in wide.columns:
            if b == (K + 1): lab = "3+"
            elif b == -(K + 1): lab = "-3+"
            else: lab = str(b)
            rename_cols.append((metric, lab))
        wide.columns = pd.MultiIndex.from_tuples(rename_cols)

        # flatten to single-level names: acc_1_c, n_-2_o, etc.
        flat = wide.copy()
        flat.columns = [f"{m}_{b}{suffix}" for (m, b) in flat.columns]

        # pair bal_acc for +/-1,2,3 (no 3+ requested here)
        def _pair(acc_pos, acc_neg):
            return (acc_pos + acc_neg) / 2

        flat[f"bal_acc_1{suffix}"] = _pair(flat.get(f"acc_1{suffix}"),  flat.get(f"acc_-1{suffix}"))
        flat[f"bal_acc_2{suffix}"] = _pair(flat.get(f"acc_2{suffix}"),  flat.get(f"acc_-2{suffix}"))
        flat[f"bal_acc_3{suffix}"] = _pair(flat.get(f"acc_3{suffix}"),  flat.get(f"acc_-3{suffix}"))
        flat[f"bal_acc_3p{suffix}"] = _pair(flat.get(f"acc_3+{suffix}"), flat.get(f"acc_-3+{suffix}"))

        return flat

    base = df_daily[[date_col, close_col] + [f"Return_{r}" for r in returns]].copy()

    by_r = {}

    for r in returns:
        ret_col = f"Return_{r}"
        df_r = _add_streak(base, ret_col)

        # merge with perf
        perf_r = perf_df[perf_df["horizon"] == r]
        if perf_filter is not None:
            perf_r = perf_filter(perf_r)

        d = df_r.merge(perf_r, on=date_col, how="inner")

        d["y"] = d[ret_col].astype("int8")
        d["p"] = d["pred"].astype(float)
        d["yhat"] = np.where(
        d["p"] >= min_th, 1,
        np.where(d["p"] <= (1 - min_th), 0, np.nan)
        )

        d = d.dropna(subset=["yhat"]).copy()

        other = perf_r.copy()
        # bucket both contexts
        d["bucket_c"] = _bucketize(d["streak"])       # close-context
        d["bucket_o"] = _bucketize(d["streak_lag1"])  # open-context
        
        #other = d[['Date', "bucket_c"]]

        tab_c = _make_context_table(d, "bucket_c", "_c")
        tab_o = _make_context_table(d, "bucket_o", "_o")

        # combine side-by-side
        out = tab_c.join(tab_o, how="outer")

        # compute wba_close / wba_open from each context’s pair balances
        out["wba_close"] = (
            w[1]   * out["bal_acc_1_c"] +
            w[2]   * out["bal_acc_2_c"] +
            w[3]   * out["bal_acc_3_c"] +
            w["3+"] * out["bal_acc_3p_c"]
        ) / max_score

        out["wba_open"] = (
            w[1]   * out["bal_acc_1_o"] +
            w[2]   * out["bal_acc_2_o"] +
            w[3]   * out["bal_acc_3_o"] +
            w["3+"] * out["bal_acc_3p_o"]
        ) / max_score

        out["wba_close"] = out["wba_close"].round(2)
        out["wba_open"]  = out["wba_open"].round(2)

        # optional: keep horizon as a column too (handy for later concat)
        out = out.reset_index()
        out.insert(0, "horizon", r)

        by_r[r] = out

    all_out = pd.concat(by_r.values(), ignore_index=True)
    return by_r, all_out, other

def bucket_scores(df_daily, perf_df, returns, min_th):

    # ---- usage ----
    by_r, all_out, d = flip_bucket_tables_multi_dual(
        df_daily=df_daily,
        perf_df=perf_df,
        returns=returns,
        min_th=min_th,
        K=3,
    )

    # horizon 10 table
    by_r[10].sort_values(["wba_close", "wba_open"], ascending=False).head(25)

    # all horizons combined
    perf_columns = ['horizon', 'model', 'train_years', 'feature_set', 'pi_size', 'min_feats',
                    'n_1_c', 'n_-1_c', 'acc_1_c', 'acc_-1_c',
                    'bal_acc_1_o', 'bal_acc_2_o', 'bal_acc_3_o', 'bal_acc_3p_o', 'wba_open', 'wba_close']

    # top per horizon (ranked by MCC desc, then Brier asc)
    top_by_horizon = (
        all_out
        .sort_values(["horizon", 'wba_close', 'wba_open'], ascending=[True, False, False])
        .groupby("horizon", as_index=False, sort=False)
        .head(10)
    )

    perf_columns = ['horizon', 'model', 'train_years', 'feature_set', 'pi_size', 'min_feats',
                'n_1_c', 'n_-1_c', 'acc_1_c', 'acc_-1_c',
                 'bal_acc_1_o', 'bal_acc_2_o', 'bal_acc_3_o', 'bal_acc_3p_o', 'wba_open', 'wba_close']

    return top_by_horizon[perf_columns].round(2)