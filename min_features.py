import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", module="joblib")
import databento as db
import exchange_calendars as xcals

def min_features():

    def close_times():

        # NYSE calendar
        cal = xcals.get_calendar("XNYS")

        # Build schedule for the date range you care about
        #start = "2018-01-01"
        #end = "2030-01-01"
        sched = cal.schedule.loc[:, ["open", "close"]].copy()

        # Convert to America/New_York
        sched["open_et"]  = sched["open"].dt.tz_convert("America/New_York")
        sched["close_et"] = sched["close"].dt.tz_convert("America/New_York")

        # Indicators
        #sched["is_trading_day"] = True
        sched["is_early_close"] = sched["close_et"].dt.time < pd.Timestamp("16:00", tz="America/New_York").time()

        # If you want a per-day close time (minutes since midnight ET)
        sched["session_duration"] = (sched["close_et"].dt.hour * 60 + sched["close_et"].dt.minute) - 9.5 * 60

        # Join to your intraday df by session date
        # assumes df has a Date column that is the NYSE session date (ET)
        sched_out = sched.reset_index().rename(columns={"index": "Date"})
        close_times_df = sched_out
        
        return close_times_df[['Date', 'close_et', 'is_early_close', 'session_duration']]

    def add_intraday_labels(df: pd.DataFrame, dt_col: str = "datetime_est") -> pd.DataFrame:
        
        out = df.copy()

        # Ensure datetime
        out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
        if out[dt_col].isna().any():
            bad = out[dt_col].isna().sum()
            raise ValueError(f"{bad} rows in {dt_col} could not be parsed to datetime.")

        # Extract time-of-day in minutes since midnight (ET)
        tod_minutes = out[dt_col].dt.hour * 60 + out[dt_col].dt.minute
        out["_tod_minutes"] = tod_minutes

        # Open time (09:30 ET) in minutes
        premarket_min = 7 * 60  # 420
        open_min = 9 * 60 + 30  # 570
        #close_min = 16 * 60 - 1   # 960

        # Minutes since open (can be negative pre-market, positive post-open)
        out["time_to_open"] = out["_tod_minutes"] - open_min
        out["time_to_close"] = (open_min + out["session_duration"]) - out["_tod_minutes"]
        
        out["session_simple"] = "post_market"

        out.loc[out["_tod_minutes"] < premarket_min, "session_simple"] = "overnight"

        out.loc[
            (out["_tod_minutes"] >= premarket_min) & (out["_tod_minutes"] < open_min),
            "session_simple"
        ] = "pre_market"

        out.loc[
            (out["_tod_minutes"] >= open_min) &
            (out["_tod_minutes"] <= open_min + out["session_duration"]),
            "session_simple"
        ] = "open_market"

        # Column 2: detailed session label (your buckets)
        out["session_detail"] = np.select(
            [
                # Pre-market buckets
                (out["_tod_minutes"] < 7 *60),
                (out["_tod_minutes"] >= 7*60) & (out["_tod_minutes"] < 9*60),
                (out["_tod_minutes"] >= 9*60) & (out["_tod_minutes"] < open_min),

                # Open market buckets
                (out["_tod_minutes"] >= open_min) & (out["_tod_minutes"] < 9*60+45),
                (out["_tod_minutes"] >= 9*60+45) & (out["_tod_minutes"] < 10*60),
                (out["_tod_minutes"] >= 10*60) & (out["_tod_minutes"] < 12*60),
                (out["_tod_minutes"] >= 12*60) & (out["_tod_minutes"] < 14*60),
                (out["_tod_minutes"] >= 14*60) & (out["_tod_minutes"] < 15*60+30),
                (out["_tod_minutes"] >= 15*60+30) & (out["_tod_minutes"] < 15*60+45),
                (out["_tod_minutes"] >= 15*60+45) & (out["_tod_minutes"] <= (open_min + out["session_duration"])),

                # Post-market buckets
                (out["_tod_minutes"] > (open_min + out["session_duration"])) & (out["_tod_minutes"] < 16*60+15),
                (out["_tod_minutes"] >= 16*60+15) & (out["_tod_minutes"] < 17*60),
                (out["_tod_minutes"] >= 17*60) & (out["_tod_minutes"] <= 20*60),
            ],
            [
                "overnight",
                "early_pre_market",
                "late_pre_market",
                "early_open",
                "late_open",
                "morning",
                "midday",
                "late_day",
                "early_close",
                "late_close",
                "early_post_market",
                "late_post_market",
                "post_market_other",
            ],
            default="other"
        )

        # Cleanup
        out = out.drop(columns=["_tod_minutes", "_detail_simple_check"], errors="ignore")
        return out

    # Read the DBN file into a DBNStore object
    dbn_store = db.DBNStore.from_file('qqq_1m.dbn')
    # Convert the data to a pandas DataFrame for analysis
    df = dbn_store.to_df()
    df_main = df.reset_index()[['symbol', 'ts_event', 'close', 'open', 'high', 'low', 'volume']].copy()

    # Add in session duration to account for early close on holidays
    df_main['datetime_est'] = (df_main['ts_event'].dt.tz_convert('America/New_York'))
    df_close_times = close_times()
    # Merge close times with intraday data
    df_main['Date'] = pd.to_datetime(df_main['datetime_est']).dt.strftime('%Y-%m-%d')
    df_close_times["Date"] = pd.to_datetime(df_close_times["Date"]).dt.date
    df_main["Date"] = pd.to_datetime(df_main["Date"]).dt.date
    df_main = df_main.merge(df_close_times[['Date', 'session_duration']], on="Date", how="left")

    df_intraday_labels = add_intraday_labels(df_main)
    df_intraday_labels['Date'] = pd.to_datetime(df_intraday_labels['datetime_est']).dt.strftime('%Y-%m-%d')
    df_labeled_final = df_intraday_labels[['symbol', 'datetime_est', 'time_to_open', 'time_to_close', 'session_simple', 
                        'session_detail', 'close', 'open', 'high', 'low', 'Date', 'session_duration', 'volume']].copy()

    df_features = df_labeled_final.dropna().copy()
    df_features = df_features[df_features['session_duration'] == 390] # filter our half days for now
    # OC, HL, CH, CL ratios and magnitudes
    o, h, l, c = (df_features[k].to_numpy() for k in ("open", "high", "low", "close"))

    def dir_th(a, b, pct):
        return (a > (1 + pct) * b).astype(np.int8) - (a < (1 - pct) * b).astype(np.int8)

    pairs = {
        "OC": (c, o),
        "HL": (h, l),
        "HC": (h, c),
        "LC": (c, l),
    }

    for k, (a, b) in pairs.items():
        #df_features[f"{k}_Minute_Direction"] = np.sign(a - b).astype(np.int8)
        df_features[f"{k}_Minute_Magnitude"] = np.round(a / b - 1, 4)
        df_features[f"{k}_Minute_Direction_Low_TH"] = dir_th(a, b, 0.001)
        df_features[f"{k}_Minute_Direction_High_TH"] = dir_th(a, b, 0.01)

    # Percent of winning and losing minutes per session_simple and session_detail?
    def percent_direction_counts(df, column_to_count, column_to_group):

        df_counts = (
            df
            .assign(
                up   = (df[column_to_count] > 0),
                down = (df[column_to_count] < 0),
                none = (df[column_to_count] == 0),
            )
            .groupby(["Date", column_to_group], sort=False)
            .agg(
                up_minutes=("up", "sum"),
                down_minutes=("down", "sum"),
                none_minutes=("none", "sum"),
            )
        ).reset_index()

        total = df_counts["up_minutes"] + df_counts["down_minutes"]

        df_counts["%_up_minutes"]   = df_counts["up_minutes"]   / total
        df_counts["%_down_minutes"] = df_counts["down_minutes"] / total
        df_counts["%_none_minutes"] = df_counts["none_minutes"] / (total + df_counts["none_minutes"])
        # keep only percent features

        counts = df_counts[
            ["Date", column_to_group, "%_up_minutes", "%_down_minutes", "%_none_minutes"]
        ]

        # pivot wide
        wide = (
            counts
            .pivot(
                index="Date",
                columns=column_to_group,
                values=["%_up_minutes", "%_down_minutes", "%_none_minutes"],
            )
        )

        # flatten MultiIndex columns and prefix with group value
        wide.columns = [
            f"{group}_{metric}"
            for metric, group in wide.columns
        ]

        return wide.fillna(-1).round(3).reset_index()

    # Average Volatility between HL 
    def magnitude_averages(df, column_to_count='OC_Minute_Magnitude', column_to_group='session_detail'):

        x = df[column_to_count].to_numpy(copy=False)

        df_tmp = df[["Date", column_to_group]].copy()
        df_tmp["pos_val"] = np.where(x > 0, x, np.nan)
        df_tmp["neg_val"] = np.where(x < 0, x, np.nan)

        agg = (
            df_tmp
            .groupby(["Date", column_to_group], sort=False)
            .agg(
                oc_pos_avg=("pos_val", "mean"),
                oc_pos_max=("pos_val", "max"),
                oc_neg_min=("neg_val", "min"),
                oc_neg_avg=("neg_val", "mean"),
            )
        )

        wide = agg.unstack(column_to_group)

        # flatten columns: <group>_<metric>
        wide.columns = [f"{grp}_{metric}" for metric, grp in wide.columns]

        return wide.fillna(0).round(6).reset_index()

    column_to_count = 'OC_Minute_Magnitude'
    column_to_group = 'session_simple'
    session_simple_counts = percent_direction_counts(df_features, column_to_count, column_to_group)

    column_to_count = 'OC_Minute_Magnitude'
    column_to_group = 'session_detail'
    session_detail_counts = percent_direction_counts(df_features, column_to_count, column_to_group)

    session_averages = magnitude_averages(df_features)

    #session_counts_final = pd.merge(session_simple_counts, session_detail_counts, how='inner', on='Date')
    session_counts_final = pd.merge(session_detail_counts, session_averages, how='inner', on='Date')

    def intraday_aggregations(df, interval, col, name):

        # ensure datetime index
        mask = (df[f"{col}"] >= 0) & (df[f"{col}"] < interval)

        daily_max_min = (
            df.loc[mask]
            .groupby("Date")["close"]
            .agg(lambda x: x.max() / x.min())
            .rename(f"max_min_{name}-{interval}m")
        )
        daily_max_min = pd.DataFrame(daily_max_min).reset_index()

        return daily_max_min

    df = df_features.copy()
    df_ph = pd.DataFrame()
    intervals = [5, 10, 15, 30, 60]
    columns = ['time_to_open', 'time_to_close']
    names = ['first', 'last']

    for interval in intervals:

        for col, name in zip(columns, names):

            daily_max_min = intraday_aggregations(df, interval, col, name)

            if df_ph.empty:
                df_ph = daily_max_min.copy()
            else:
                df_ph = df_ph.merge(daily_max_min, how="left", on="Date")

    df_maxmin = df_ph.copy()

    df_features_final = pd.merge(session_counts_final, df_maxmin, how='inner', on='Date')

    close_cols = df_features_final.columns[(df_features_final.columns.str.contains("close_")) | 
                                           (df_features_final.columns.str.contains("post_")) | 
                                           (df_features_final.columns.str.contains("overnight_"))].to_list()
    ave_cols = (
        df_features_final
        .loc[:, ~df_features_final.columns.isin(close_cols)]  # drop close_ columns
        .iloc[:, 1:]                               # drop first column
        .columns
        .to_list()
    )
    cols = [c for c in ave_cols if c != "Date"]

    df_features_final[[f"{c}_roll3" for c in cols]] = (
        df_features_final[cols]
        .rolling(window=3, min_periods=1)   # use min_periods=3 if you want NaN until 3 days exist
        .mean()
    )

    return df_features_final