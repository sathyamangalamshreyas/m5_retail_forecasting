import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pd

    from pathlib import Path


@app.cell
def _():
    mo.md(r"""
    # M5 Forecasting — exploratory data analysis

    This notebook explores the M5 bottom-level item–store series, calendar features,
    demand intermittency, and weekly selling prices.

    **Data-handling choices**
    - `sales_train_evaluation.csv` is the sales source. It contains `d_1`–`d_1941`,
      including the 28-day validation target window (`d_1914`–`d_1941`,
    2016-04-25 through 2016-05-22).
    - `sales_train_validation.csv` is not reloaded: Phase 0 already verified that its
      daily values exactly match `d_1`–`d_1913` in the evaluation file.
    - Daily sales remain in wide form. This notebook does not melt the 30,490 series
      into a 59-million-row table.
    - Price coverage is checked against weekly calendar keys. Price records are not
      expanded to daily rows.

    Run from the project root with:

    ```bash
    pixi run -e eda marimo edit notebooks/01_m5_eda.py
    ```
    """)
    return


@app.cell
def _():
    def locate_project_root(start: Path) -> Path:
        for candidate in (start, *start.parents):
            if (candidate / "pixi.toml").is_file():
                return candidate
        raise FileNotFoundError(
            "Could not locate pixi.toml. Start Marimo from the M5 project directory."
        )

    project_root = locate_project_root(Path.cwd())
    raw_dir = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    required_files = [
        raw_dir / "sales_train_evaluation.csv",
        raw_dir / "calendar.csv",
        raw_dir / "sell_prices.csv",
    ]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Required M5 files were not found:\n{missing_text}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    mo.md(
        f"""
        **Project root:** `{project_root}`  
        **Raw data:** `{raw_dir}`
        """
    )
    return raw_dir, reports_dir


@app.cell
def _(raw_dir):
    sales_path = raw_dir / "sales_train_evaluation.csv"
    calendar_path = raw_dir / "calendar.csv"
    prices_path = raw_dir / "sell_prices.csv"

    # Discover the actual daily columns from the header, then read them compactly.
    sales_header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    metadata_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    if sales_header[: len(metadata_columns)] != metadata_columns:
        raise ValueError(
            "Unexpected M5 sales metadata schema. Expected the six standard metadata columns first."
        )
    day_columns = [column for column in sales_header if column.startswith("d_")]
    expected_days = [f"d_{day}" for day in range(1, 1942)]
    if day_columns != expected_days:
        raise ValueError("Expected contiguous sales columns d_1 through d_1941.")

    sales_dtypes = {
        "id": "string",
        "item_id": "category",
        "dept_id": "category",
        "cat_id": "category",
        "store_id": "category",
        "state_id": "category",
    }
    sales_dtypes.update({column: "uint16" for column in day_columns})
    sales = pd.read_csv(sales_path, dtype=sales_dtypes, low_memory=False)

    calendar = pd.read_csv(
        calendar_path,
        dtype={
            "d": "string",
            "wm_yr_wk": "int32",
            "weekday": "category",
            "wday": "int8",
            "month": "int8",
            "year": "int16",
            "event_name_1": "category",
            "event_type_1": "category",
            "event_name_2": "category",
            "event_type_2": "category",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
        },
        low_memory=False,
    )
    calendar["date"] = pd.to_datetime(calendar["date"])

    sell_prices = pd.read_csv(
        prices_path,
        dtype={
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int32",
            "sell_price": "float32",
        },
        low_memory=False,
    )

    load_summary = pd.DataFrame(
        {
            "file": [sales_path.name, calendar_path.name, prices_path.name],
            "rows": [len(sales), len(calendar), len(sell_prices)],
            "columns": [sales.shape[1], calendar.shape[1], sell_prices.shape[1]],
        }
    )
    mo.md("### Loaded source files")
    load_summary
    return calendar, day_columns, metadata_columns, sales, sell_prices


@app.cell
def _(calendar, day_columns, metadata_columns, sales):
    # These checks repeat only the lightweight structural assertions needed by this notebook.
    sales_metadata = sales[metadata_columns].copy()
    sales_values = sales[day_columns].to_numpy(copy=False)
    n_series, n_days = sales_values.shape

    calendar_sales = calendar.iloc[:n_days].copy()
    expected_day_labels = [f"d_{day}" for day in range(1, n_days + 1)]
    if calendar_sales["d"].tolist() != expected_day_labels:
        raise ValueError("Calendar row order does not match sales day-column order.")
    if sales_metadata[["item_id", "store_id"]].duplicated().any():
        raise ValueError("Duplicate item-store series found in sales metadata.")

    mo.md(
        f"""
        ### Sales matrix dimensions

        - **Bottom-level series:** {n_series:,}
        - **Daily observations per series:** {n_days:,}
        - **Sales matrix dtype:** `{sales_values.dtype}`
        - **Approximate daily matrix size:** {sales_values.nbytes / (1024**2):,.1f} MiB
        - **Date range:** {calendar_sales['date'].min():%Y-%m-%d} to {calendar_sales['date'].max():%Y-%m-%d}
        """
    )
    return calendar_sales, sales_metadata, sales_values


@app.cell
def _(sales_metadata, sales_values):
    # Compute series features directly from the wide NumPy matrix.
    active_days = np.count_nonzero(sales_values, axis=1)
    total_units = sales_values.sum(axis=1, dtype=np.uint64)
    sum_of_squares = np.einsum(
        "ij,ij->i",
        sales_values,
        sales_values,
        dtype=np.uint64,
        casting="unsafe",
        optimize=True,
    )

    mean_nonzero = np.divide(
        total_units,
        active_days,
        out=np.full(len(active_days), np.nan, dtype=np.float64),
        where=active_days > 0,
    )
    mean_square_nonzero = np.divide(
        sum_of_squares,
        active_days,
        out=np.full(len(active_days), np.nan, dtype=np.float64),
        where=active_days > 0,
    )
    # E[x^2] - E[x]^2 gives population variance over non-zero demand days.
    variance_nonzero = np.maximum(mean_square_nonzero - np.square(mean_nonzero), 0)
    cv2_nonzero = np.divide(
        variance_nonzero,
        np.square(mean_nonzero),
        out=np.full(len(active_days), np.nan, dtype=np.float64),
        where=mean_nonzero > 0,
    )

    series_summary = sales_metadata.copy()
    series_summary["total_units"] = total_units.astype(np.int64)
    series_summary["mean_daily_units"] = total_units / sales_values.shape[1]
    series_summary["active_days"] = active_days
    series_summary["zero_day_pct"] = 100 * (1 - active_days / sales_values.shape[1])
    series_summary["mean_nonzero_demand"] = mean_nonzero
    series_summary["ADI"] = np.divide(
        sales_values.shape[1],
        active_days,
        out=np.full(len(active_days), np.nan, dtype=np.float64),
        where=active_days > 0,
    )
    series_summary["CV2"] = cv2_nonzero

    # Attach the canonical date range as notebook metadata, not as a melted sales table.
    series_summary.index.name = "sales_row"
    overall_summary = pd.DataFrame(
        {
            "measure": [
                "Item-store series",
                "Total units sold",
                "Series with at least one sale",
                "Completely zero series",
                "Mean zero-sales days per series",
                "Median ADI among selling series",
                "Median CV² among selling series",
            ],
            "value": [
                len(series_summary),
                int(total_units.sum()),
                int((active_days > 0).sum()),
                int((active_days == 0).sum()),
                float(series_summary["zero_day_pct"].mean()),
                float(series_summary["ADI"].median()),
                float(series_summary["CV2"].median()),
            ],
        }
    )

    mo.vstack([
        mo.md(
        """
        ### Series-level demand features

        - `ADI` is the number of observed days divided by the number of selling days.
        - `CV2` is the squared coefficient of variation of **non-zero** daily demand.
        - These are descriptive features. We will use their distributions to understand
          intermittency rather than imposing demand-class thresholds at this stage.
        """
    ),
        overall_summary.round(2)
    ])
    return (series_summary,)


@app.cell
def _(series_summary):
    def summarize_by(group_column: str) -> pd.DataFrame:
        return (
            series_summary.groupby(group_column, observed=True, sort=True)
            .agg(
                series_count=("id", "size"),
                total_units=("total_units", "sum"),
                mean_units_per_series=("total_units", "mean"),
                mean_zero_day_pct=("zero_day_pct", "mean"),
                median_ADI=("ADI", "median"),
                median_CV2=("CV2", "median"),
            )
            .sort_values("total_units", ascending=False)
        )

    state_summary = summarize_by("state_id")
    store_summary = summarize_by("store_id")
    category_summary = summarize_by("cat_id")
    department_summary = summarize_by("dept_id")

    mo.vstack([
        mo.md("### Sales composition by state"),
        state_summary.round(2)
    ])
    return category_summary, department_summary, store_summary


@app.cell
def _(category_summary, department_summary, store_summary):
    mo.md("### Sales composition by store, category, and department")
    mo.vstack(
        [
            mo.md("**Stores**"),
            store_summary.round(2).sort_index(),
            mo.md("**Categories**"),
            category_summary.round(2),
            mo.md("**Departments**"),
            department_summary.round(2),
        ]
    )
    return


@app.cell
def _(calendar_sales, sales_values):
    daily_total_units = sales_values.sum(axis=0, dtype=np.uint64)
    daily_totals = calendar_sales[
        ["d", "date", "weekday", "wday", "month", "year", "wm_yr_wk"]
    ].copy()
    daily_totals["total_units"] = daily_total_units.astype(np.int64)
    daily_totals["rolling_7d_mean"] = daily_totals["total_units"].rolling(7).mean()

    fig_daily, ax_daily = plt.subplots(figsize=(12, 4))
    ax_daily.plot(daily_totals["date"], daily_totals["total_units"], label="Daily units", alpha=0.65)
    ax_daily.plot(
        daily_totals["date"],
        daily_totals["rolling_7d_mean"],
        label="7-day rolling mean",
        linewidth=2,
    )
    ax_daily.set(title="Total daily unit sales across all 30,490 series", xlabel="Date", ylabel="Units")
    ax_daily.legend()
    ax_daily.grid(True, alpha=0.25)
    plt.tight_layout()

    mo.md("### Aggregate demand over time")
    fig_daily
    return (daily_totals,)


@app.cell
def _(daily_totals):
    weekday_profile = (
        daily_totals.groupby(["wday", "weekday"], observed=True)["total_units"]
        .mean()
        .reset_index()
        .sort_values("wday")
    )
    monthly_profile = (
        daily_totals.groupby(["year", "month"], observed=True)["total_units"]
        .mean()
        .reset_index()
    )
    fig_weekday, ax_weekday = plt.subplots(figsize=(8, 4))
    ax_weekday.bar(weekday_profile["weekday"].astype(str), weekday_profile["total_units"])
    ax_weekday.set(title="Mean total sales by weekday", xlabel="Weekday", ylabel="Mean daily units")
    ax_weekday.tick_params(axis="x", rotation=30)
    ax_weekday.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    fig_month, ax_month = plt.subplots(figsize=(10, 4))
    month_labels = monthly_profile["year"].astype(str) + "-" + monthly_profile["month"].astype(str).str.zfill(2)
    ax_month.plot(month_labels, monthly_profile["total_units"], marker="o")
    ax_month.set(title="Mean daily sales by calendar month", xlabel="Month", ylabel="Mean daily units")
    ax_month.tick_params(axis="x", rotation=60)
    ax_month.grid(True, alpha=0.25)
    plt.tight_layout()

    mo.md("### Weekly and monthly patterns")
    mo.vstack([fig_weekday, fig_month])
    return


@app.cell
def _(calendar_sales, daily_totals):
    event_columns = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    calendar_features = calendar_sales[["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year", *event_columns, "snap_CA", "snap_TX", "snap_WI"]].copy()
    calendar_features["has_event"] = calendar_features[["event_name_1", "event_name_2"]].notna().any(axis=1)
    calendar_features["total_units"] = daily_totals["total_units"].to_numpy()

    event_day_summary = (
        calendar_features.groupby("has_event", observed=True)["total_units"]
        .agg(["count", "mean", "median", "std"])
        .rename(index={False: "No named event", True: "Named event"})
    )
    event_counts = pd.concat(
        [
            calendar_features["event_name_1"].dropna().astype(str),
            calendar_features["event_name_2"].dropna().astype(str),
        ],
        ignore_index=True,
    ).value_counts().head(15).rename_axis("event_name").to_frame("calendar_occurrences")

    snap_summary = pd.DataFrame(
        {
            "state_id": ["CA", "TX", "WI"],
            "SNAP_days_in_history": [
                int(calendar_features["snap_CA"].sum()),
                int(calendar_features["snap_TX"].sum()),
                int(calendar_features["snap_WI"].sum()),
            ],
            "history_days": [len(calendar_features)] * 3,
        }
    )
    snap_summary["SNAP_day_pct"] = 100 * snap_summary["SNAP_days_in_history"] / snap_summary["history_days"]

    mo.md("### Calendar events")
    mo.vstack([event_day_summary.round(2), event_counts])
    return calendar_features, snap_summary


@app.cell
def _(calendar_features, sales_metadata, sales_values, snap_summary):
    state_positions = sales_metadata.groupby("state_id", observed=True).indices
    state_daily_data = {
        state: sales_values[positions, :].sum(axis=0, dtype=np.uint64)
        for state, positions in state_positions.items()
    }
    state_daily = pd.DataFrame(state_daily_data)
    state_daily.insert(0, "date", calendar_features["date"].to_numpy())

    fig_state_snap, ax_state_snap = plt.subplots(figsize=(12, 4))
    for state in ["CA", "TX", "WI"]:
        ax_state_snap.plot(state_daily["date"], state_daily[state], label=f"{state} sales")
    ax_state_snap.set(title="Daily unit sales by state", xlabel="Date", ylabel="Units")
    ax_state_snap.legend()
    ax_state_snap.grid(True, alpha=0.25)
    plt.tight_layout()

    mo.md("### State-level demand and SNAP calendar flags")
    mo.vstack([snap_summary.round(2), fig_state_snap])
    return


@app.cell
def _(series_summary):
    selling_series = series_summary.loc[series_summary["active_days"] > 0]

    fig_zero_days, ax_zero_days = plt.subplots(figsize=(8, 4))
    ax_zero_days.hist(selling_series["zero_day_pct"], bins=30)
    ax_zero_days.set(title="Distribution of zero-sales days by item–store series", xlabel="Zero-sales days (%)", ylabel="Series count")
    ax_zero_days.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    fig_adi, ax_adi = plt.subplots(figsize=(8, 4))
    finite_adi = selling_series["ADI"].replace([np.inf, -np.inf], np.nan).dropna()
    ax_adi.hist(finite_adi.clip(upper=finite_adi.quantile(0.99)), bins=40)
    ax_adi.set(title="ADI distribution (upper tail clipped at the 99th percentile)", xlabel="ADI", ylabel="Series count")
    ax_adi.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    fig_cv2, ax_cv2 = plt.subplots(figsize=(8, 4))
    finite_cv2 = selling_series["CV2"].replace([np.inf, -np.inf], np.nan).dropna()
    ax_cv2.hist(finite_cv2.clip(upper=finite_cv2.quantile(0.99)), bins=40)
    ax_cv2.set(title="CV² of non-zero demand (upper tail clipped at the 99th percentile)", xlabel="CV²", ylabel="Series count")
    ax_cv2.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    mo.md(
        """
        ### Demand intermittency and variability

        Read the zero-day share, ADI, and CV² distributions together. A series can have
        frequent sales but highly variable non-zero quantities, or infrequent sales with
        relatively stable non-zero quantities. Keep those patterns distinct for later
        baseline selection and segmented evaluation.
        """
    )
    mo.vstack([fig_zero_days, fig_adi, fig_cv2])
    return


@app.cell
def _(calendar_sales, sales_values, series_summary):
    # Select deterministic examples to inspect visibly different demand profiles.
    highest_total_index = series_summary["total_units"].idxmax()
    median_total_index = (series_summary["total_units"] - series_summary["total_units"].median()).abs().idxmin()
    intermittent_candidates = series_summary.loc[series_summary["active_days"] >= 5]
    most_intermittent_index = intermittent_candidates["ADI"].idxmax()
    sample_indices = [highest_total_index, median_total_index, most_intermittent_index]
    sample_labels = ["Highest total units", "Near-median total units", "Most intermittent (at least 5 selling days)"]

    sample_series_info = series_summary.loc[sample_indices, ["id", "item_id", "store_id", "cat_id", "dept_id", "total_units", "active_days", "zero_day_pct", "ADI", "CV2"]].copy()
    sample_series_info.insert(0, "selection", sample_labels)

    sample_figures = []
    date_values = calendar_sales["date"].to_numpy()
    for row_index, label in zip(sample_indices, sample_labels):
        fig_sample, ax_sample = plt.subplots(figsize=(11, 3))
        ax_sample.plot(date_values, sales_values[row_index, :], linewidth=1)
        series_id = str(series_summary.loc[row_index, "id"])
        ax_sample.set(title=f"{label}: {series_id}", xlabel="Date", ylabel="Units")
        ax_sample.grid(True, alpha=0.25)
        plt.tight_layout()
        sample_figures.append(fig_sample)

    mo.md("### Representative bottom-level series")
    mo.vstack([sample_series_info.round(2), *sample_figures])
    return


@app.cell
def _(calendar, sales_metadata, sell_prices):
    # Aggregate prices at the weekly item-store grain; no daily price expansion.
    price_series_summary = (
        sell_prices.groupby(["store_id", "item_id"], observed=True, sort=False)
        .agg(
            price_weeks=("wm_yr_wk", "nunique"),
            unique_price_points=("sell_price", "nunique"),
            min_price=("sell_price", "min"),
            median_price=("sell_price", "median"),
            max_price=("sell_price", "max"),
            first_price_week=("wm_yr_wk", "min"),
            last_price_week=("wm_yr_wk", "max"),
        )
        .reset_index()
    )
    price_series_summary["price_changed_at_least_once"] = price_series_summary["unique_price_points"] > 1

    # Check future weekly price availability over both the known validation window and
    # the calendar-only competition test window. Expected grid size is small (~few 100k).
    future_calendar = calendar.loc[calendar["d"].str.extract(r"d_(\d+)", expand=False).astype(int) >= 1914, ["d", "wm_yr_wk"]].copy()
    future_calendar["day_number"] = future_calendar["d"].str.extract(r"d_(\d+)", expand=False).astype(int)
    future_calendar["period"] = np.where(future_calendar["day_number"] <= 1941, "validation_d1914_d1941", "competition_test_d1942_d1969")
    future_week_periods = future_calendar[["period", "wm_yr_wk"]].drop_duplicates()
    bottom_level_keys = sales_metadata[["store_id", "item_id"]].drop_duplicates()
    price_key_columns = sell_prices[["store_id", "item_id", "wm_yr_wk"]]

    coverage_parts = []
    for period_name, period_weeks in future_week_periods.groupby("period", observed=True):
        week_frame = period_weeks[["wm_yr_wk"]].drop_duplicates()
        expected_grid = bottom_level_keys.merge(week_frame, how="cross")
        checked_grid = expected_grid.merge(
            price_key_columns,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
            indicator=True,
        )
        period_coverage = (
            checked_grid.groupby("wm_yr_wk", observed=True)['_merge']
            .apply(lambda values: (values == "both").sum())
            .rename("series_with_price")
            .reset_index()
        )
        period_coverage["expected_series"] = len(bottom_level_keys)
        period_coverage["coverage_pct"] = 100 * period_coverage["series_with_price"] / len(bottom_level_keys)
        period_coverage["period"] = period_name
        coverage_parts.append(period_coverage)
    future_price_coverage = pd.concat(coverage_parts, ignore_index=True)

    price_pair_summary = pd.DataFrame(
        {
            "measure": [
                "Weekly price rows",
                "Item-store pairs with at least one price",
                "Item-store pairs with a price change",
                "Median distinct price weeks per pair",
            ],
            "value": [
                len(sell_prices),
                len(price_series_summary),
                int(price_series_summary["price_changed_at_least_once"].sum()),
                float(price_series_summary["price_weeks"].median()),
            ],
        }
    )

    fig_price_changes, ax_price_changes = plt.subplots(figsize=(8, 4))
    ax_price_changes.hist(price_series_summary["unique_price_points"], bins=30)
    ax_price_changes.set(title="Distinct observed price values per item–store pair", xlabel="Distinct price values", ylabel="Item–store pairs")
    ax_price_changes.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    mo.md(
        """
        ### Weekly selling prices

        Price keys are `(store_id, item_id, wm_yr_wk)`. The table is kept weekly; the
        calendar supplies the mapping from day to selling week. Coverage below is a
        diagnostic only: missing future price keys should not be silently forward-filled.
        """
    )
    mo.vstack([price_pair_summary, future_price_coverage, fig_price_changes])
    return


@app.cell
def _(series_summary):
    from skimpy import skim

    mo.md("### Compact statistical profile with Skimpy")
    skim_columns = [
        "total_units",
        "mean_daily_units",
        "active_days",
        "zero_day_pct",
        "mean_nonzero_demand",
        "ADI",
        "CV2",
    ]
    skim(series_summary[skim_columns].sample(n=min(10_000, len(series_summary)), random_state=42))
    return


@app.cell
def _(reports_dir, series_summary):
    # Optional heavier HTML profile. Set this to True only when you want to generate it.
    RUN_HTML_PROFILE = True
    profile_output = None
    if RUN_HTML_PROFILE:
        from data_profiling import ProfileReport

        profile_sample = series_summary.sample(n=min(10_000, len(series_summary)), random_state=42)
        profile = ProfileReport(
            profile_sample,
            title="M5 item-store series profile (10k-row sample)",
            minimal=True,
        )
        profile_output = reports_dir / "m5_series_profile.html"
        profile.to_file(str(profile_output))

    if RUN_HTML_PROFILE:
        mo.md(f"HTML profile saved to `{profile_output}`")
    else:
        mo.md(
            "**HTML profiling is disabled by default.** Set `RUN_HTML_PROFILE = True` in this cell to create `reports/m5_series_profile.html` with `fg-data-profiling`."
        )
    return


@app.cell
def _(series_summary):
    # D-Tale starts a local web server. Keep it opt-in rather than launching it on notebook load.
    RUN_DTALE = False
    dtale_instance = None
    if RUN_DTALE:
        import dtale

        dtale_sample = series_summary.sample(n=min(5_000, len(series_summary)), random_state=42)
        dtale_instance = dtale.show(
            dtale_sample,
            host="127.0.0.1",
            port=40000,
            open_browser=False,
            ignore_duplicate=True,
        )
        dtale_link = dtale_instance._main_url
        mo.md(f"D-Tale is running locally: `{dtale_link}`")
    else:
        mo.md(
            "**D-Tale is disabled by default.** Set `RUN_DTALE = True` in this cell to launch a local D-Tale view of a 5,000-series sample."
        )
    return


@app.cell
def _():
    mo.md(r"""
    s## EDA checkpoint — questions to answer before modeling

    Record observations from the tables and plots before proceeding:

    1. **Demand shape:** How many series are sparse or highly intermittent? Are there
       meaningful differences by state, store, category, or department?
    2. **Seasonality:** Is the aggregate weekly pattern strong? Are there visible
       longer-term or calendar-month changes?
    3. **Events and SNAP:** Do named-event days or SNAP-flagged periods coincide with
       aggregate changes? Treat these as associations, not causal effects.
    4. **Prices:** What proportion of validation and competition-test weeks have
       complete item–store price coverage? Which series have little price variation?
    5. **Modeling implications:** Which baseline families should be included for dense,
       intermittent, and low-volume series? Which calendar and price covariates are
       genuinely known for every forecast origin?

    **Next step:** use these findings to design the stratified pilot and baseline
    comparison. Avoid selecting a single representative series or one aggregation
    level as a proxy for the full M5 distribution.
    """)
    return


@app.cell
def _(series_summary):
    def _():
        import numpy as np
        import pandas as pd

        # Define classification function based on standard thresholds
        def classify_demand(row):
            adi = row['ADI']
            cv2 = row['CV2']
    
            # Handle NaNs (e.g., completely zero series if any slip through)
            if pd.isna(adi) or pd.isna(cv2):
                return 'Unclassified / Zero'
    
            is_sporadic = adi >= 1.32
            is_variable = cv2 >= 0.49
    
            if not is_sporadic and not is_variable:
                return 'Smooth'
            elif is_sporadic and not is_variable:
                return 'Intermittent'
            elif not is_sporadic and is_variable:
                return 'Erratic'
            else:
                return 'Lumpy'

        # Apply the classification to your series_summary dataframe
        series_summary['demand_category'] = series_summary.apply(classify_demand, axis=1)

        # Check the distribution of categories across your portfolio
        category_counts = series_summary['demand_category'].value_counts(normalize=True) * 100
        return print(category_counts.round(2))


    _()
    return


@app.cell
def _(series_summary):
    def _():
        import marimo as mo
        import pandas as pd

        def calculate_demand_distribution(series_summary):
            """Calculates the demand classification breakdown and portfolio metrics."""
            df = series_summary.copy()
    
            # 1. Classification logic (Syntetos-Boylan thresholds)
            def classify_demand(row):
                adi = row['ADI']
                cv2 = row['CV2']
                if pd.isna(adi) or pd.isna(cv2):
                    return 'Unclassified / Zero'
        
                is_sporadic = adi >= 1.32
                is_variable = cv2 >= 0.49
        
                if not is_sporadic and not is_variable:
                    return 'Smooth'
                elif is_sporadic and not is_variable:
                    return 'Intermittent'
                elif not is_sporadic and is_variable:
                    return 'Erratic'
                else:
                    return 'Lumpy'
            
            df['demand_category'] = df.apply(classify_demand, axis=1)
    
            # 2. Aggregate metrics by category
            summary = df.groupby('demand_category').agg(
                series_count=('ADI', 'count'),
                total_units=('total_units', 'sum'),
                mean_adi=('ADI', 'mean'),
                mean_cv2=('CV2', 'mean')
            ).reset_index()
    
            # 3. Add percentage of total portfolio
            total_series = len(df)
            summary['portfolio_pct'] = (summary['series_count'] / total_series) * 100
    
            # 4. Clean column names and sort logically
            category_order = ['Smooth', 'Intermittent', 'Erratic', 'Lumpy', 'Unclassified / Zero']
            summary['demand_category'] = pd.Categorical(summary['demand_category'], categories=category_order, ordered=True)
            summary = summary.sort_values('demand_category').reset_index(drop=True)
    
            summary = summary.rename(columns={
                'demand_category': 'Demand Category',
                'series_count': 'Item Count',
                'portfolio_pct': 'Portfolio (%)',
                'total_units': 'Total Units Sold',
                'mean_adi': 'Mean ADI',
                'mean_cv2': 'Mean CV²'
            })
    
            return summary

        # Generate the distribution summary table
        distribution_df = calculate_demand_distribution(series_summary)

        # Display using marimo's interactive UI table with rounded metrics
        return mo.vstack([
            mo.md(
                """
                ### Portfolio Demand Classification Breakdown
                The table below groups your item-store series into the 4 standard demand patterns, 
                showing their share of the portfolio and total volume contribution.
                """
            ),
            mo.ui.table(distribution_df.round(2))
        ])


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
