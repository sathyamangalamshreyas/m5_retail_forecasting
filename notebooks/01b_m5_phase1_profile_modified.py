import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import os
    import warnings
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl


@app.cell
def _():
    mo.md(r"""
    # M5 Phase 1 — profile the subset before STUMPY

    Builds on `01_m5_eda.py` (aggregate metrics) and answers the questions that decide
    how STUMPY and Timestral should be used:

    1. **Launches and lifecycle** — which series are live, since when, and which have died?
    2. **Intermittency, live-adjusted** — ADI/CV² and Syntetos–Boylan classes computed only
       over the days an item was actually on sale, compared with the naive full-history values.
    3. **Concentration** — how much of the M5 evaluation weight sits in how many series.
    4. **Is the subset representative?** — subset vs. all stores.
    5. **Calendar effects** — weekday, annual shape, events, SNAP, against a trend baseline.
    6. **Price effects** — price dynamics and a within-series elasticity estimate.
    7. **STUMPY readiness** — how many z-normalised windows are constant (all zeros).

    **Leakage rule:** every statistic uses only the training window `d_1 … d_(N-28)`.
    The last 28 days (the validation target) are never looked at here.

    Output: `artifacts/phase1_series_table.parquet`, one row per subset series, used later to
    stratify samples and select STUMPY inputs.

    Runs in either Pixi environment (polars + numpy + matplotlib only):

    ```bash
    pixi run phase1-notebook
    ```
    """)
    return


@app.cell
def _():
    SUBSET = os.environ.get("M5_SUBSET", "ca_1").lower()  # override per run: M5_SUBSET=wi_3
    HOLDOUT = 28  # trailing days never used for any statistic
    MIN_HISTORY = 56  # live days below which a series counts as "short history"
    ADI_CUT, CV2_CUT = 1.32, 0.49  # Syntetos-Boylan-Croston thresholds
    CLASSES = ["Smooth", "Erratic", "Intermittent", "Lumpy", "No sales"]
    return ADI_CUT, CLASSES, CV2_CUT, HOLDOUT, MIN_HISTORY, SUBSET


@app.cell
def _(HOLDOUT, SUBSET):
    def _find_root(start: Path) -> Path:
        for candidate in (start, *start.parents):
            if (candidate / "pixi.toml").is_file():
                return candidate
        raise FileNotFoundError("pixi.toml not found; start marimo inside the project.")

    ROOT = _find_root(Path.cwd())
    PROC = ROOT / "data" / "processed"
    SUB = PROC / "subsets" / SUBSET
    _required = [
        PROC / "calendar.parquet",
        PROC / "aggregates.parquet",
        PROC / "sell_prices.parquet",
        SUB / "sales.npy",
        SUB / "price.npy",
        SUB / "series.parquet",
    ]
    _missing = [str(p) for p in _required if not p.is_file()]
    if _missing:
        raise FileNotFoundError("Run `pixi run prep-data` first. Missing:\n" + "\n".join(_missing))

    sales_all = np.load(SUB / "sales.npy")
    price_all = np.load(SUB / "price.npy")
    series = pl.read_parquet(SUB / "series.parquet")
    cal = pl.read_parquet(PROC / "calendar.parquet")
    agg = pl.read_parquet(PROC / "aggregates.parquet")

    n_series, n_days = sales_all.shape
    assert series.height == n_series
    T = n_days - HOLDOUT  # training days
    sales = sales_all[:, :T]
    price = price_all[:, :T]
    cal_n = cal.head(n_days)

    dates = cal_n["date"].to_numpy()[:T]
    wday = cal_n["wday"].to_numpy()[:T].astype(int)
    month = cal_n["month"].to_numpy()[:T].astype(int)
    year = cal_n["year"].to_numpy()[:T].astype(int)
    wk_id = cal_n["wm_yr_wk"].to_numpy()[:T]
    dec25 = ((cal_n["date"].dt.month() == 12) & (cal_n["date"].dt.day() == 25)).to_numpy()[:T]
    wday_names = dict(cal_n.select("wday", "weekday").unique().sort("wday").iter_rows())
    subset_stores = series["store_id"].unique().sort().to_list()

    mo.md(
        f"""
        **Subset `{SUBSET}`** — stores {subset_stores}: {n_series:,} series × {n_days:,} days.
        Training window: d_1 … d_{T} ({dates[0]} → {dates[-1]}); held out: last {HOLDOUT} days.
        """
    )
    return (
        PROC,
        ROOT,
        T,
        agg,
        cal,
        cal_n,
        dates,
        dec25,
        month,
        n_series,
        price,
        sales,
        series,
        subset_stores,
        wday,
        wday_names,
        wk_id,
        year,
    )


@app.cell
def _(T, agg, cal_n, dec25):
    def get_agg(level: str, key: str) -> np.ndarray:
        """Training-window aggregate series (all stores) from aggregates.parquet."""
        row = agg.filter((pl.col("level") == level) & (pl.col("key") == key))
        return np.asarray(row["sales"][0].to_list(), dtype=np.float64)[:T]

    _ev1 = cal_n["event_name_1"].to_list()[:T]
    _ev2 = cal_n["event_name_2"].to_list()[:T]
    _ty1 = cal_n["event_type_1"].to_list()[:T]
    _ty2 = cal_n["event_type_2"].to_list()[:T]
    event_days: dict[str, list[int]] = {}
    event_type: dict[str, str] = {}
    for _i, (_a, _b, _ta, _tb) in enumerate(zip(_ev1, _ev2, _ty1, _ty2)):
        for _name, _typ in ((_a, _ta), (_b, _tb)):
            if _name is not None:
                event_days.setdefault(_name, []).append(_i)
                event_type.setdefault(_name, _typ)
    is_event = np.zeros(T, dtype=bool)
    for _idx in event_days.values():
        is_event[_idx] = True

    def centered_nanmean(x, valid, window):
        k = np.ones(window)
        num = np.convolve(np.where(valid, x, 0.0), k, mode="same")
        den = np.convolve(valid.astype(float), k, mode="same")
        return num / np.maximum(den, 1.0)

    def seasonal_ratio(x, wday, trend_exclude, wf_exclude):
        """x / (35-day centred mean x weekday factor).

        Returns (ratio, weekday_factors[7], trend, weekly_strength). The 35-day mean
        spans exactly five weeks so weekday effects cancel in the trend.
        """
        trend = centered_nanmean(x, ~trend_exclude, 35)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = x / trend
        ok = ~wf_exclude & np.isfinite(r)
        wf = np.array([r[(wday == k) & ok].mean() for k in range(1, 8)])
        wf = wf / wf.mean()
        ratio = r / wf[wday - 1]
        resid = r[ok] - wf[wday[ok] - 1]
        strength = max(0.0, 1.0 - resid.var() / r[ok].var())
        return ratio, wf, trend, strength

    skip_days = dec25 | is_event
    return (
        centered_nanmean,
        event_days,
        event_type,
        get_agg,
        is_event,
        seasonal_ratio,
        skip_days,
    )


@app.cell
def _():
    mo.md(r"""
    ## 1 · Launches and lifecycle

    `first_live_d` is the first day with a price entry. Zeros before it mean "not on sale
    yet", not zero demand. `first_live_d == 1` is left-censored: live since at least d_1.
    """)
    return


@app.cell
def _(MIN_HISTORY, T, dates, n_series, price, sales, series):
    _fl = series["first_live_d"].fill_null(T + 1).to_numpy().astype(np.int64)
    live_start0 = np.clip(_fl - 1, 0, T)  # metadata launch index; 0-based
    # Daily availability is authoritative: an item can have multiple live blocks.
    live_mask = np.isfinite(price)
    live_days = live_mask.sum(axis=1)
    left_censored = live_mask[:, 0]
    never_live = live_days == 0

    nz = sales > 0
    has_sale = nz.any(axis=1)
    trailing_zero_run = np.zeros(n_series, dtype=np.int32)
    for i in range(n_series):
        live_idx = np.flatnonzero(live_mask[i])
        if live_idx.size:
            last_live = int(live_idx[-1])
            # Restrict the trailing run to the final contiguous live block.
            final_block_start = last_live
            while final_block_start > 0 and live_mask[i, final_block_start - 1]:
                final_block_start -= 1
            prior_sales = np.flatnonzero(nz[i, final_block_start : last_live + 1])
            trailing_zero_run[i] = last_live - (final_block_start + int(prior_sales[-1])) if prior_sales.size else last_live - final_block_start + 1

    def _longest_zero_run(live_nz):
        if live_nz.size == 0:
            return 0
        z = np.concatenate(([0], (~live_nz).astype(np.int8), [0]))
        d = np.diff(z)
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return int((ends - starts).max()) if starts.size else 0

    max_zero_run = np.zeros(n_series, dtype=np.int32)
    zero_run_tail_p_upper = np.ones(n_series, dtype=np.float64)
    for i in range(n_series):
        # Evaluate each contiguous price-available block separately; gaps are not
        # treated as observed zero-demand days.
        starts = np.flatnonzero(live_mask[i] & ~np.r_[False, live_mask[i, :-1]])
        ends = np.flatnonzero(live_mask[i] & ~np.r_[live_mask[i, 1:], False]) + 1
        block_tail_probs = []
        for a, b in zip(starts, ends):
            block = nz[i, a:b]
            n_live, n_sales = block.size, int(block.sum())
            run = _longest_zero_run(block)
            max_zero_run[i] = max(max_zero_run[i], run)
            if run > 0 and n_live > 0:
                p_sale = n_sales / n_live
                # Conservative union bound for at least one run of `run` zeros
                # under an iid Bernoulli reference with this block's sale rate.
                block_tail_probs.append(min(1.0, (n_live - run + 1) * (1.0 - p_sale) ** run))
        zero_run_tail_p_upper[i] = min(1.0, sum(block_tail_probs)) if block_tail_probs else 1.0
    # Benjamini-Hochberg FDR screen across series; this is a diagnostic flag,
    # not a stockout label (the iid reference is deliberately simple).
    order = np.argsort(zero_run_tail_p_upper)
    ranks = np.arange(1, n_series + 1)
    bh = zero_run_tail_p_upper[order] * n_series / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    zero_run_q = np.ones(n_series, dtype=np.float64)
    zero_run_q[order] = np.minimum(bh, 1.0)
    unusual_zero_run = zero_run_q <= 0.05
    short_history = live_days < MIN_HISTORY
    sales_outside_live = int((nz & ~live_mask).sum())

    def _row(label, mask):
        return {"measure": label, "series": int(mask.sum()), "share_%": round(100 * mask.mean(), 1)}

    lifecycle = pl.DataFrame(
        [
            _row("Live since d_1 (left-censored)", left_censored),
            _row("Launched inside training window", ~left_censored & ~never_live),
            _row("Not live in training window", never_live),
            _row(f"Short history (< {MIN_HISTORY} live days)", short_history & ~never_live),
            _row("No sales at all in training window", ~has_sale),
            _row("Trailing zeros ≥ 28 days", trailing_zero_run >= 28),
            _row("Trailing zeros ≥ 56 days", trailing_zero_run >= 56),
            _row("Longest zero run while live ≥ 28 days", max_zero_run >= 28),
            _row("Unusually long zero run (BH q ≤ 0.05)", unusual_zero_run),
        ]
    )

    _launch_d = _fl[~left_censored & ~never_live]
    fig_launch, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    _a1.hist(_launch_d, bins=50)
    _a1.set(title="Launch day of items that started after d_1", xlabel="day index", ylabel="series")
    _a2.plot(dates, live_mask.sum(axis=0))
    _a2.set(title="Series live over time", ylabel="live series")
    for _a in (_a1, _a2):
        _a.grid(alpha=0.25)
    plt.tight_layout()

    mo.vstack(
        [
            lifecycle,
            mo.md(
                f"Sales on days with no price entry: **{sales_outside_live:,}** (expected 0). "
                f"Unusually long zero runs under the within-series sale-rate reference: **{int(unusual_zero_run.sum()):,} series**. "
                "This conservative iid-based screen is an investigation flag, not evidence of stockouts."
            ),
            fig_launch,
        ]
    )
    return (
        has_sale,
        live_days,
        live_mask,
        live_start0,
        max_zero_run,
        never_live,
        short_history,
        trailing_zero_run,
        unusual_zero_run,
        zero_run_q,
        zero_run_tail_p_upper,
    )


@app.cell
def _():
    mo.md(r"""
    ## 2 · Intermittency: naive vs. live-adjusted

    ADI = live days / selling days. CV² = squared coefficient of variation of non-zero
    demand (population variance, as in `01_m5_eda.py`). "Live-adjusted" counts only days
    with an actual price record, including any separate live blocks. Thresholds: ADI ≥ 1.32
    and CV² ≥ 0.49 (SBC).
    """)
    return


@app.cell
def _(ADI_CUT, CLASSES, CV2_CUT, live_mask, sales, series):
    def _demand_stats(x, mask):
        active = (x > 0) & mask
        n_obs = mask.sum(axis=1)
        n_act = active.sum(axis=1)
        xf = np.where(active, x, 0).astype(np.float64)
        s, ss = xf.sum(axis=1), (xf**2).sum(axis=1)
        denom = np.maximum(n_act, 1)
        adi = np.where(n_act > 0, n_obs / denom, np.nan)
        mean = s / denom
        var = np.maximum(ss / denom - mean**2, 0)
        cv2 = np.where(n_act > 0, var / np.where(mean > 0, mean**2, 1.0), np.nan)
        return adi, cv2, n_act

    def _sbc(adi, cv2, n_act):
        ok = n_act > 0
        sporadic, variable = adi >= ADI_CUT, cv2 >= CV2_CUT
        return np.select(
            [
                ok & ~sporadic & ~variable,
                ok & ~sporadic & variable,
                ok & sporadic & ~variable,
                ok & sporadic & variable,
            ],
            CLASSES[:4],
            default="No sales",
        )

    adi_naive, cv2_naive, nact_naive = _demand_stats(sales, np.ones_like(live_mask))
    adi_live, cv2_live, nact_live = _demand_stats(sales, live_mask)
    class_naive = _sbc(adi_naive, cv2_naive, nact_naive)
    class_live = _sbc(adi_live, cv2_live, nact_live)

    _order = {c: i for i, c in enumerate(CLASSES)}
    transition = (
        pl.DataFrame({"naive": class_naive.tolist(), "live": class_live.tolist()})
        .group_by("naive", "live")
        .len()
        .pivot(on="live", index="naive", values="len")
        .fill_null(0)
        .with_columns(_o=pl.col("naive").replace_strict(_order, default=99))
        .sort("_o")
        .drop("_o")
    )
    transition = transition.select("naive", *[c for c in CLASSES if c in transition.columns])

    _by_cat = (
        pl.DataFrame({"cat_id": series["cat_id"], "class": class_live.tolist()})
        .group_by("cat_id", "class")
        .len()
        .with_columns((100 * pl.col("len") / pl.col("len").sum().over("cat_id")).round(1).alias("share_%"))
        .pivot(on="class", index="cat_id", values="share_%")
        .fill_null(0.0)
        .sort("cat_id")
    )

    _changed = float(np.mean(class_naive != class_live))
    fig_sbc, _axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for _ax, _adi, _cv2, _title in (
        (_axes[0], adi_naive, cv2_naive, "Naive (all training days)"),
        (_axes[1], adi_live, cv2_live, "Live-adjusted"),
    ):
        _ok = np.isfinite(_adi) & np.isfinite(_cv2)
        _ax.hexbin(np.log10(_adi[_ok]), np.clip(_cv2[_ok], 0, 3), gridsize=40, bins="log")
        _ax.axvline(np.log10(ADI_CUT), color="r", lw=1)
        _ax.axhline(CV2_CUT, color="r", lw=1)
        _ax.set(title=_title, xlabel="log10 ADI", ylabel="CV² (clipped at 3)")
    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"**{100 * _changed:.1f}%** of series change class when the pre-launch zeros "
                "are removed. Rows: naive class; columns: live-adjusted class."
            ),
            transition,
            mo.md("**Live-adjusted class mix by category (% of series)**"),
            _by_cat,
            fig_sbc,
        ]
    )
    return adi_live, adi_naive, class_live, class_naive, cv2_live, cv2_naive


@app.cell
def _(CLASSES, class_live, n_series, price, sales):
    mo.md(r"""
    ## 3 · Concentration of the evaluation weight

    M5's WRMSSE weights each series by its dollar sales over the last 28 *training* days.
    Here that is computed within the subset (the official weights use all 42,840 series).
    """)
    rev28 = np.nansum(sales[:, -28:] * price[:, -28:], axis=1)
    units28 = sales[:, -28:].sum(axis=1)
    weight = rev28 / rev28.sum()
    _sorted = np.sort(weight)[::-1]
    _cum = np.cumsum(_sorted)
    _n = {f"{int(q * 100)}%": int(np.searchsorted(_cum, q)) + 1 for q in (0.5, 0.8, 0.9, 0.99)}

    _by_class = (
        pl.DataFrame({"class": class_live.tolist(), "units28": units28, "rev": weight})
        .group_by("class")
        .agg(
            pl.len().alias("series"),
            pl.col("units28").sum().alias("units28"),
            pl.col("rev").sum().alias("rev_share"),
        )
        .with_columns(
            (100 * pl.col("series") / n_series).round(1).alias("series_%"),
            (100 * pl.col("units28") / pl.col("units28").sum()).round(1).alias("units_%"),
            (100 * pl.col("rev_share")).round(1).alias("revenue_weight_%"),
        )
        .select("class", "series", "series_%", "units_%", "revenue_weight_%")
        .with_columns(_o=pl.col("class").replace_strict({c: i for i, c in enumerate(CLASSES)}, default=99))
        .sort("_o")
        .drop("_o")
    )

    fig_conc, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.plot(100 * np.arange(1, n_series + 1) / n_series, 100 * _cum)
    _ax.set(
        title="Cumulative revenue weight (last 28 training days)",
        xlabel="% of series (largest first)",
        ylabel="% of revenue",
    )
    _ax.grid(alpha=0.25)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                "Series needed to reach 50 / 80 / 90 / 99% of revenue weight: "
                + ", ".join(f"**{v:,}** ({k})" for k, v in _n.items())
            ),
            _by_class,
            fig_conc,
        ]
    )
    return rev28, units28, weight


@app.cell
def _(PROC, T, get_agg, sales, subset_stores, wday):
    mo.md(r"""
    ## 4 · Is the subset representative?

    Compares the subset with every store using the all-store aggregates (training window):
    share of units, category mix, weekday profile, and the shape of the 28-day level.
    """)
    _cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    _grand = get_agg("total", "total").sum()
    _sub_total = sales.sum(axis=0).astype(np.float64)

    def _wd_profile(x):
        p = np.array([x[wday == k].mean() for k in range(1, 8)])
        return p / p.mean()

    def _level(x):
        return np.convolve(x, np.ones(28) / 28, mode="valid")

    _sub_wd, _sub_lv = _wd_profile(_sub_total), _level(_sub_total)
    _wide = pl.read_parquet(PROC / "sales_wide.parquet")
    _store_keys = sorted(_wide["store_id"].unique().to_list())
    _rows = []
    for _k in _store_keys:
        _x = get_agg("store_id", _k)
        _row = {"store": _k, "in_subset": _k in subset_stores, "units_share_%": round(100 * _x.sum() / _grand, 1)}
        for _c in _cats:
            _row[f"{_c}_%"] = round(100 * get_agg("store_cat", f"{_k}|{_c}").sum() / _x.sum(), 1)
        _row["weekday_profile_corr"] = round(float(np.corrcoef(_sub_wd, _wd_profile(_x))[0, 1]), 3)
        _row["level_28d_corr"] = round(float(np.corrcoef(_sub_lv, _level(_x))[0, 1]), 3)
        _rows.append(_row)
    store_compare = pl.DataFrame(_rows)

    _qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    _day_cols = [f"d_{i}" for i in range(1, T + 1)]
    _all_train = _wide.select(_day_cols).to_numpy()
    _zs = 1.0 - (_all_train > 0).mean(axis=1)
    _in = _wide["store_id"].is_in(subset_stores).to_numpy()
    zero_compare = pl.DataFrame(
        {
            "quantile": _qs,
            "zero_share_subset": np.round(np.quantile(_zs[_in], _qs), 3),
            "zero_share_all": np.round(np.quantile(_zs, _qs), 3),
        }
    )

    mo.vstack(
        [
            store_compare,
            mo.md(
                "Zero share per series (training window only, same d_1..d_T range for "
                "subset and full portfolio):"
            ),
            zero_compare,
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 5 · Calendar effects against a trend baseline

    Each series is divided by `35-day centred mean × weekday factor`. Dec 25 (stores closed)
    is excluded from the trend, and event days and Dec 25 from the weekday factors, so
    events don't contaminate their own baseline. Effects are **associations**, estimated from
    ~5 occurrences per event; treat small effects with caution.
    """)
    return


@app.cell
def _(dec25, get_agg, seasonal_ratio, skip_days, wday, wday_names, year):
    _cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    _levels = {"TOTAL": get_agg("total", "total"), **{c: get_agg("cat_id", c) for c in _cats}}
    _res = {k: seasonal_ratio(v, wday, dec25, skip_days) for k, v in _levels.items()}

    wd_table = pl.DataFrame(
        {"weekday": [wday_names[k] for k in range(1, 8)], **{k: np.round(r[1], 3) for k, r in _res.items()}}
    )
    strength_table = pl.DataFrame(
        {"level": list(_res), "weekly_seasonal_strength": [round(float(r[3]), 3) for r in _res.values()]}
    )

    _years = np.unique(year)
    _tot = _levels["TOTAL"]
    _rows = []
    for _y in _years:
        _m = year == _y
        _rows.append({"year": int(_y), "days": int(_m.sum()), "mean_daily_units": round(float(_tot[_m].mean()), 0)})
    yearly = pl.DataFrame(_rows).with_columns(
        (100 * (pl.col("mean_daily_units") / pl.col("mean_daily_units").shift(1) - 1)).round(1).alias("yoy_%")
    )

    fig_wd, _ax = plt.subplots(figsize=(7, 3.4))
    for _k, _r in _res.items():
        _ax.plot(range(1, 8), _r[1], marker="o", label=_k)
    _ax.set_xticks(range(1, 8), [wday_names[k][:3] for k in range(1, 8)])
    _ax.set(title="Weekday factors (1.0 = average day)")
    _ax.legend()
    _ax.grid(alpha=0.25)
    plt.tight_layout()

    mo.vstack(
        [
            mo.hstack([wd_table, strength_table, yearly], justify="start", gap=2),
            fig_wd,
            mo.md(
                "Yearly rows: first and last years are partial; compare complete years only. "
                "`weekly_seasonal_strength` = 1 − var(residual)/var(ratio): "
                "close to 1 means the weekday pattern dominates short-term variation."
            ),
        ]
    )
    return


@app.cell
def _(
    T,
    centered_nanmean,
    dec25,
    get_agg,
    month,
    seasonal_ratio,
    skip_days,
    wday,
):
    _tot = get_agg("total", "total")
    _, _, _trend, _ = seasonal_ratio(_tot, wday, dec25, skip_days)
    _annual = centered_nanmean(_tot, ~dec25, 365)
    _lo, _hi = 182, T - 182
    _rel = (_trend / _annual)[_lo:_hi]
    _mth = month[_lo:_hi]
    annual_profile = np.array([_rel[_mth == m].mean() for m in range(1, 13)])

    fig_annual, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.bar(range(1, 13), annual_profile - 1)
    _ax.set(
        title="Annual shape of total sales (35-day mean ÷ 365-day mean − 1)",
        xlabel="month",
        ylabel="relative level",
    )
    _ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig_annual
    return


@app.cell
def _(T, cal_n, live_mask, sales):
    mo.md(r"""
    ### Like-for-like trend: series live at d_1

    This separates changes in demand for the opening assortment from growth caused by
    newly launched items. The cohort is fixed using price availability on d_1; no holdout
    observations are used.
    """)
    _cohort = live_mask[:, 0]
    _all = sales.sum(axis=0).astype(np.float64)
    _like = sales[_cohort].sum(axis=0).astype(np.float64)
    _dates = cal_n["date"].to_numpy()[:T]
    _smooth_all = np.convolve(_all, np.ones(35) / 35, mode="valid")
    _smooth_like = np.convolve(_like, np.ones(35) / 35, mode="valid")
    _all_index = 100 * _smooth_all / max(_smooth_all[0], 1e-12)
    _like_index = 100 * _smooth_like / max(_smooth_like[0], 1e-12)
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.plot(_dates[17 : T - 17], _all_index, label="All series (includes launches)")
    _ax.plot(_dates[17 : T - 17], _like_index, label=f"Live at d_1 cohort ({int(_cohort.sum()):,} series)")
    _ax.set(title="35-day smoothed sales: total vs. fixed opening cohort", ylabel="index (first 35-day window = 100)")
    _ax.legend()
    _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    dates,
    dec25,
    event_days: dict[str, list[int]],
    event_type: dict[str, str],
    get_agg,
    is_event,
    seasonal_ratio,
    skip_days,
    wday,
):
    _cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    _r = {"total": seasonal_ratio(get_agg("total", "total"), wday, dec25, skip_days)[0]}
    _r.update({c: seasonal_ratio(get_agg("cat_id", c), wday, dec25, skip_days)[0] for c in _cats})

    _rows = []
    for _name, _idx in event_days.items():
        _window = np.zeros(len(dates), dtype=bool)
        for _day in _idx:
            _window[max(0, _day - 7) : min(len(dates), _day + 8)] = True
        _row = {"event": _name, "type": event_type[_name], "occurrences": len(_idx), "event_window_days": int(_window.sum())}
        for _k, _v in _r.items():
            _row[f"{_k}_%"] = round(100 * (float(_v[_window].mean()) - 1), 1)
        _rows.append(_row)
    event_table = (
        pl.DataFrame(_rows)
        .with_columns(pl.col("total_%").abs().alias("_abs"))
        .sort("_abs", descending=True)
        .drop("_abs")
    )

    _d25 = np.flatnonzero(dec25)
    xmas = pl.DataFrame(
        {
            "date": [str(dates[i])[:10] for i in _d25],
            "total_units": [int(get_agg("total", "total")[i]) for i in _d25],
            "ratio_to_baseline_%": [round(100 * float(_r["total"][i]), 1) for i in _d25],
        }
    )
    mo.vstack(
        [
            mo.md(
                f"**Named events** ({len(event_days)} distinct, {int(is_event.sum())} event days). "
                "Percent difference vs. the weekday-and-trend baseline, all stores:"
            ),
            event_table,
            mo.md("**Dec 25 closure** — near-zero total sales across all stores:"),
            xmas,
        ]
    )
    return


@app.cell
def _(T, agg, cal_n, dec25, seasonal_ratio, skip_days, wday):
    _cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    _store_cat = {
        k: np.asarray(s, dtype=np.float64)[:T]
        for k, s in agg.filter(pl.col("level") == "store_cat").select("key", "sales").iter_rows()
    }
    _states = sorted({k.split("_")[0] for k in _store_cat})
    _rows = []
    for _st in _states:
        _flag = cal_n[f"snap_{_st}"].to_numpy()[:T].astype(bool)
        for _c in _cats:
            _x = sum(v for k, v in _store_cat.items() if k.startswith(f"{_st}_") and k.endswith(f"|{_c}"))
            _ratio = seasonal_ratio(_x, wday, dec25, skip_days)[0]
            _use = ~skip_days
            _eff = _ratio[_use & _flag].mean() / _ratio[_use & ~_flag].mean() - 1
            _rows.append({"state": _st, "category": _c, "snap_days": int(_flag.sum()), "snap_effect_%": round(100 * float(_eff), 1)})
    snap_table = pl.DataFrame(_rows)
    _controls = snap_table.filter(pl.col("category").is_in(["HOBBIES", "HOUSEHOLD"]))
    _control_mean = _controls.group_by("state").agg(pl.col("snap_effect_%").mean().alias("nonfood_control_%"))
    snap_table = snap_table.join(_control_mean, on="state", how="left").with_columns(
        (pl.col("snap_effect_%") - pl.col("nonfood_control_%")).alias("above_nonfood_control_pp")
    )

    fig_snap, _ax = plt.subplots(figsize=(7, 3.4))
    _w = 0.25
    for _j, _c in enumerate(_cats):
        _sub = snap_table.filter(pl.col("category") == _c)
        _ax.bar(np.arange(len(_states)) + _j * _w, _sub["snap_effect_%"].to_numpy(), _w, label=_c)
    _ax.set_xticks(np.arange(len(_states)) + _w, _states)
    _ax.set(title="Sales on SNAP days vs. non-SNAP days (%)", ylabel="% difference")
    _ax.legend()
    _ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    mo.vstack([mo.md("**SNAP effect** by state × category (all stores in the state). HOBBIES/HOUSEHOLD mean is a negative-control reference; the FOODS-minus-control column is descriptive, not a causal estimate:"), snap_table, fig_snap])
    return


@app.cell
def _():
    mo.md(r"""
    ## 6 · Price effects

    Weekly grain (prices only change weekly); only complete 7-day weeks inside the training
    window. The elasticity estimate regresses `log(1 + weekly units)` on
    `log(price / series median price)` with series and week fixed effects, so it measures
    how a series responds when *its own* price moves, net of store-wide weekly shocks.
    Prices are not randomised (promotions coincide with events), so read it as an association.
    """)
    return


@app.cell
def _(T, price, sales, wk_id):
    _change = np.flatnonzero(np.diff(wk_id) != 0) + 1
    _starts = np.concatenate(([0], _change))
    _lens = np.diff(np.concatenate((_starts, [T])))
    _keep = _lens == 7
    weekly_units = np.add.reduceat(sales, _starts, axis=1)[:, _keep].astype(np.float64)
    weekly_price = price[:, _starts[_keep]].astype(np.float64)
    weekly_live = ~np.isnan(weekly_price)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _med = np.nanmedian(weekly_price, axis=1)
        _pmax, _pmin = np.nanmax(weekly_price, axis=1), np.nanmin(weekly_price, axis=1)
    rel_price = weekly_price / _med[:, None]

    _both = weekly_live[:, 1:] & weekly_live[:, :-1]
    _dp = np.where(_both, weekly_price[:, 1:] / weekly_price[:, :-1] - 1, np.nan)
    _changed = _both & (np.abs(_dp) > 1e-6)
    _abs = np.abs(_dp[_changed])
    _discount = weekly_live & (rel_price < 0.95)

    price_dynamics = pl.DataFrame(
        {
            "measure": [
                "Complete weeks analysed",
                "Live series-weeks",
                "Week-to-week price changes (% of consecutive live pairs)",
                "Median |price change| when it changes (%)",
                "Series with ≥ 1 price change (%)",
                "Live series-weeks priced > 5% below series median (%)",
            ],
            "value": [
                f"{int(_keep.sum()):,}",
                f"{int(weekly_live.sum()):,}",
                f"{100 * float(_changed.sum() / max(_both.sum(), 1)):.2f}",
                f"{100 * float(np.median(_abs)):.1f}" if _abs.size else "n/a",
                f"{100 * float(np.mean((_pmax - _pmin) > 1e-6)):.1f}",
                f"{100 * float(_discount.sum() / max(weekly_live.sum(), 1)):.1f}",
            ],
        }
    )
    has_price_variation = (_pmax - _pmin) > 1e-6
    price_dynamics
    return has_price_variation, rel_price, weekly_live, weekly_units


@app.cell
def _():
    n_boot = mo.ui.slider(start=0, stop=200, step=10, value=40, label="Bootstrap resamples (0 = point estimate only)")
    n_boot
    return (n_boot,)


@app.cell
def _(
    has_price_variation,
    n_boot,
    rel_price,
    series,
    weekly_live,
    weekly_units,
):
    def _twfe_slope(y, x, w, iters=6):
        y, x = np.where(w, y, 0.0), np.where(w, x, 0.0)
        for _ in range(iters):
            for a in (y, x):
                a -= ((a * w).sum(1) / w.sum(1).clip(1))[:, None] * w
                a -= ((a * w).sum(0) / w.sum(0).clip(1))[None, :] * w
        den = (x * x * w).sum()
        return float((x * y * w).sum() / den) if den > 0 else float("nan")

    _y = np.log1p(weekly_units)
    _x = np.log(np.where(weekly_live, rel_price, 1.0))
    _w = weekly_live
    _zero_share = ((weekly_units == 0) & _w).sum(1) / _w.sum(1).clip(1)
    _eligible = has_price_variation & (_w.sum(1) >= 26)
    _cat = series["cat_id"].to_numpy()
    _groups = {
        "all": _eligible,
        "dense (≤10% zero weeks)": _eligible & (_zero_share <= 0.10),
        **{c: _eligible & (_cat == c) for c in sorted(set(_cat))},
    }
    _rng = np.random.default_rng(0)
    _rows = []
    for _name, _sel in _groups.items():
        _idx = np.flatnonzero(_sel)
        if _idx.size < 20:
            continue
        _point = _twfe_slope(_y[_idx].copy(), _x[_idx].copy(), _w[_idx])
        _lo = _hi = float("nan")
        if n_boot.value > 0:
            _b = []
            for _ in range(n_boot.value):
                _s = _rng.choice(_idx, _idx.size, replace=True)
                _b.append(_twfe_slope(_y[_s].copy(), _x[_s].copy(), _w[_s]))
            _lo, _hi = np.nanpercentile(_b, [2.5, 97.5])
        _rows.append(
            {
                "group": _name,
                "series": int(_idx.size),
                "slope_log1p_units_per_log_price": round(_point, 3),
                "ci95_low": round(float(_lo), 3),
                "ci95_high": round(float(_hi), 3),
            }
        )
    elasticity = pl.DataFrame(_rows)
    mo.vstack(
        [
            elasticity,
            mo.md(
                "Negative = demand falls when the series' own price rises. Slopes on "
                "`log1p(units)` are attenuated for sparse series; the *dense* row is the "
                "cleanest read."
            ),
        ]
    )
    return


@app.cell
def _(PROC, T, cal, price, series, subset_stores):
    _last_live = ~np.isnan(price[:, T - 1])
    _live_keys = series.filter(pl.Series(_last_live)).select("item_id", "store_id")
    _weeks = cal.filter(pl.col("d_num") > T).select("wm_yr_wk").unique()
    _prices = pl.scan_parquet(PROC / "sell_prices.parquet").filter(pl.col("store_id").is_in(subset_stores)).collect()
    future_price_coverage = (
        _live_keys.join(_weeks, how="cross")
        .join(_prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")
        .group_by("wm_yr_wk")
        .agg(pl.col("sell_price").is_not_null().mean().alias("share_with_price"))
        .sort("wm_yr_wk")
        .with_columns(pl.col("share_with_price").round(4))
    )
    mo.vstack(
        [
            mo.md(
                f"**Known-future price coverage.** Of the {int(_last_live.sum()):,} series live on "
                f"the last training day, share with a price in each week after it (validation + "
                f"competition horizon). Prices can be supplied to Timestral as known-future "
                f"covariates only where this is ≈ 1."
            ),
            future_price_coverage,
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 7 · STUMPY readiness: constant windows

    A z-normalised matrix profile cannot compare a constant window (e.g. all zeros): it has
    no shape. For each series, the share of length-`m` windows whose every day has a
    price record that are exactly constant, for the two windows most likely to matter:
    a week (`m = 7`) and the forecast horizon (`m = 28`).
    """)
    return


@app.cell
def _(class_live, live_days, live_mask, sales):
    def _window_diagnostics(x, live, m):
        n, t = x.shape
        n_windows = t - m + 1
        if n_windows <= 0:
            raise ValueError(f"m={m} exceeds available history {t}")
        # Prefix sums make each window's non-zero count and live-day count O(1).
        nz_prefix = np.pad(np.cumsum((x > 0).astype(np.int32), axis=1), ((0, 0), (1, 0)))
        live_prefix = np.pad(np.cumsum(live.astype(np.int32), axis=1), ((0, 0), (1, 0)))
        nz_count = nz_prefix[:, m:] - nz_prefix[:, :-m]
        live_count = live_prefix[:, m:] - live_prefix[:, :-m]
        valid = live_count == m
        density = nz_count / m
        # A fully-live window is constant iff it has zero transitions.
        changes = (np.diff(x, axis=1) != 0).astype(np.int32)
        change_prefix = np.pad(np.cumsum(changes, axis=1), ((0, 0), (1, 0)))
        change_count = change_prefix[:, m - 1 :] - change_prefix[:, : n_windows]
        constant = (change_count == 0) & valid
        valid_n = valid.sum(axis=1)
        const_share = np.divide(constant.sum(axis=1), valid_n, out=np.full(n, np.nan), where=valid_n > 0)
        density_quantiles = np.full((n, 3), np.nan)
        for i in range(n):
            if valid[i].any():
                density_quantiles[i] = np.quantile(density[i, valid[i]], [0.1, 0.5, 0.9])
        density_counts = {
            k: ((nz_count >= k) & valid).sum(axis=1)
            for k in sorted({1, 2, 3, max(1, m // 4), m})
        }
        return const_share, valid_n, density_quantiles, density_counts

    const_share_7, valid_windows_7, density_q_7, density_counts_7 = _window_diagnostics(sales, live_mask, 7)
    const_share_28, valid_windows_28, density_q_28, density_counts_28 = _window_diagnostics(sales, live_mask, 28)

    _df = pl.DataFrame({"class": class_live.tolist(), "c7": const_share_7, "c28": const_share_28})        .with_columns(pl.col("c7", "c28").fill_nan(None))
    readiness = _df.group_by("class").agg(
        pl.len().alias("series"),
        pl.col("c7").median().round(3).alias("median_const_share_m7"),
        pl.col("c28").median().round(3).alias("median_const_share_m28"),
        (pl.col("c7") > 0.5).sum().alias("series_const_gt50%_m7"),
        (pl.col("c28") > 0.5).sum().alias("series_const_gt50%_m28"),
    ).sort("series", descending=True)

    # Density sensitivity: number of series with at least one valid window
    # containing >= k non-zero observations. No single threshold is assumed.
    _density_rows = []
    for _m, _counts, _valid in ((7, density_counts_7, valid_windows_7), (28, density_counts_28, valid_windows_28)):
        for _k, _arr in _counts.items():
            _density_rows.append({"window_m": _m, "min_nonzero": _k,
                "series_with_eligible_window": int(((_arr > 0) & (_valid > 0)).sum()),
                "series_with_any_valid_window": int((_valid > 0).sum())})
    density_sensitivity = pl.DataFrame(_density_rows)

    ELIG_MAX_CONST = 0.5
    ELIG_MIN_LIVE_DAYS = 365
    eligible_znorm_m28 = (live_days >= ELIG_MIN_LIVE_DAYS) & (np.nan_to_num(const_share_28, nan=1.0) <= ELIG_MAX_CONST)
    eligible_znorm_m7 = (live_days >= ELIG_MIN_LIVE_DAYS) & (np.nan_to_num(const_share_7, nan=1.0) <= ELIG_MAX_CONST)

    mo.vstack([readiness, mo.md("**Window-density sensitivity** — counts of series having at least one fully live window with the stated minimum number of non-zero days. This is descriptive; select a cutoff only after inspecting downstream motif quality."), density_sensitivity,
        mo.md(f"Initial constant-window screen: **{int(eligible_znorm_m7.sum()):,}** series for m=7 and **{int(eligible_znorm_m28.sum()):,}** for m=28. This is not a complete STUMPY-readiness decision." )])
    return const_share_28, const_share_7, density_q_28, density_q_7, eligible_znorm_m28, eligible_znorm_m7, valid_windows_28, valid_windows_7


@app.cell
def _(
    ROOT,
    adi_live,
    adi_naive,
    class_live,
    class_naive,
    const_share_28,
    const_share_7,
    cv2_live,
    cv2_naive,
    eligible_znorm_m28,
    eligible_znorm_m7,
    has_sale,
    live_days,
    max_zero_run,
    never_live,
    unusual_zero_run,
    zero_run_q,
    zero_run_tail_p_upper,
    density_q_28,
    density_q_7,
    rev28,
    series,
    short_history,
    trailing_zero_run,
    units28,
    weight,
):
    phase1_table = series.select("id", "item_id", "dept_id", "cat_id", "store_id", "first_live_d", "first_sale_d").with_columns(
        pl.Series("live_days", live_days),
        pl.Series("adi_naive", adi_naive),
        pl.Series("cv2_naive", cv2_naive),
        pl.Series("class_naive", class_naive.tolist()),
        pl.Series("adi_live", adi_live),
        pl.Series("cv2_live", cv2_live),
        pl.Series("class_live", class_live.tolist()),
        pl.Series("short_history", short_history),
        pl.Series("never_live", never_live),
        pl.Series("no_sales", ~has_sale),
        pl.Series("trailing_zero_run", trailing_zero_run),
        pl.Series("max_zero_run_live", max_zero_run),
        pl.Series("zero_run_tail_p_upper", zero_run_tail_p_upper),
        pl.Series("zero_run_q_bh", zero_run_q),
        pl.Series("unusual_zero_run_q05", unusual_zero_run),
        pl.Series("density_q10_m7", density_q_7[:, 0]),
        pl.Series("density_median_m7", density_q_7[:, 1]),
        pl.Series("density_q90_m7", density_q_7[:, 2]),
        pl.Series("density_q10_m28", density_q_28[:, 0]),
        pl.Series("density_median_m28", density_q_28[:, 1]),
        pl.Series("density_q90_m28", density_q_28[:, 2]),
        pl.Series("units_last28", units28),
        pl.Series("revenue_last28", rev28),
        pl.Series("revenue_weight", weight),
        pl.Series("const_share_m7", const_share_7),
        pl.Series("const_share_m28", const_share_28),
        pl.Series("znorm_eligible_m7", eligible_znorm_m7),
        pl.Series("znorm_eligible_m28", eligible_znorm_m28),
    )
    _out = ROOT / "artifacts" / "phase1_series_table.parquet"
    _out.parent.mkdir(parents=True, exist_ok=True)
    phase1_table.write_parquet(_out, compression="zstd")
    mo.vstack(
        [
            mo.md(f"Saved **{phase1_table.height:,}** rows × {phase1_table.width} columns to `{_out}`."),
            phase1_table.head(5),
        ]
    )
    return (phase1_table,)


@app.cell
def _(phase1_table):
    _t = phase1_table
    _n = _t.height
    _pct = lambda m: f"{100 * float(m.sum()) / _n:.1f}%"  # noqa: E731
    mo.md(
        f"""
        ## Phase 1 checkpoint

        Numbers from this run (subset of {_n:,} series, training window only):

        - **Lifecycle:** {_pct(_t['first_live_d'].eq(1))} live since d_1; {_pct(_t['short_history'])} short history; {_pct(_t['no_sales'])} no sales; {_pct(_t['trailing_zero_run'].ge(28))} end with ≥ 28 zero days.
        - **Intermittency:** {_pct(_t['class_naive'].ne(_t['class_live']))} of series change SBC class once pre-launch zeros are removed.
        - **Zero-run screen:** {_pct(_t['unusual_zero_run_q05'])} flagged by the within-series sale-rate reference (BH q ≤ 0.05); investigate, do not label as stockouts.
        - **STUMPY:** {_pct(_t['znorm_eligible_m7'])} pass the constant-window/live-history screen for m = 7; {_pct(_t['znorm_eligible_m28'])} for m = 28. Window density is reported separately and should guide method selection.

        **Questions to answer before Phase 2** (write your conclusions here):

        1. Which stratum (class × volume) gets which STUMPY treatment — raw z-norm, smoothed input, `normalize=False`, or aggregated levels only?
        2. Do the weekday factors, annual shape and event effects above justify `m = 7`, `28`, `364`?
        3. Are event and SNAP effects large enough to be worth encoding as known-future covariates *and* as annotation vectors?
        4. Is price variation / elasticity strong enough to include price as a covariate for Timestral?
        5. Is the subset representative enough to carry conclusions to the other 9 stores?
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
