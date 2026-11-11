import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import pandas as pd

    from pathlib import Path
    from joblib import Parallel, delayed
    import multiprocessing

    import matplotlib.pyplot as plt
    import seaborn as sns

    import stumpy
    from stumpy import chains


@app.cell
def _():
    mo.md(r"""
    # M5 Retail Forecasting: STUMPY Matrix Profile Explorer
    Interactive motif and anomaly detection for M5 sales time-series.
    """)
    return


@app.cell
def _():
    # --- PHASE 1: Data Ingestion & Setup ---

    data_path = Path("./data/raw/")

    sales_file = data_path / "sales_train_evaluation.csv"
    calendar_file = data_path / "calendar.csv"
    prices_file = data_path / "sell_prices.csv"

    sales_df = pd.read_csv(sales_file)
    calendar_df = pd.read_csv(calendar_file)
    prices_df = pd.read_csv(prices_file)
    return (sales_df,)


@app.cell
def _(sales_df):
    # Extract date columns (d_1 to d_1941)

    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    return (day_cols,)


@app.cell
def _(sales_df):
    # Reactive UI controls for marimo

    sample_items = sales_df["item_id"].unique()[:50].tolist()
    sample_stores = sales_df["store_id"].unique().tolist()
    
    item_dropdown = mo.ui.dropdown(options=sample_items, value=sample_items[0], label="Select Item ID")
    store_dropdown = mo.ui.dropdown(options=sample_stores, value=sample_stores[0], label="Select Store ID")
    window_slider = mo.ui.slider(start=7, stop=90, step=1, value=28, label="Matrix Profile Window (m)")
    
    mo.hstack([item_dropdown, store_dropdown, window_slider])
    return item_dropdown, store_dropdown, window_slider


@app.cell
def _(day_cols, item_dropdown, sales_df, store_dropdown):
    # Filter dataset for the selected item and store

    selected_item = item_dropdown.value
    selected_store = store_dropdown.value

    series_row = sales_df[
        (sales_df["item_id"] == selected_item) & 
        (sales_df["store_id"] == selected_store)
    ]

    if series_row.empty:
        sales_ts = []
        item_meta = {}
    else:
        # Extract sales values as float64 array required by STUMPY
        sales_ts = series_row[day_cols].values.flatten().astype(float)
        item_meta = series_row.iloc[0][["item_id", "store_id", "cat_id", "dept_id"]].to_dict()
    return item_meta, sales_ts, selected_item, selected_store


@app.cell
def _(item_meta, sales_ts):
    # Display basic info about the selected series
    mo.md(
        f"""
        **Active Series Overview:**
        - **Item:** `{item_meta.get('item_id', 'N/A')}`
        - **Category:** `{item_meta.get('cat_id', 'N/A')} / {item_meta.get('dept_id', 'N/A')}`
        - **Store:** `{item_meta.get('store_id', 'N/A')}`
        - **Total Timesteps:** `{len(sales_ts)}` days
        """
    )
    return


@app.cell
def _(sales_ts, window_slider):
    m = window_slider.value

    if len(sales_ts) > m * 2:
        # Compute Matrix Profile (MP) and Matrix Profile Index (MPI)
        mp_result = stumpy.stump(sales_ts, m=m)
        mp = mp_result[:, 0]
        mpi = mp_result[:, 1]
    
        # Identify top Motif (most recurring pattern: lowest distance)
        motif_idx = np.argsort(mp)[0]
    
        # Identify top Discord (most anomalous pattern: highest distance)
        discord_idx = np.argsort(mp)[-1]
    else:
        mp, mpi, motif_idx, discord_idx = [], [], None, None
    return discord_idx, m, motif_idx, mp


@app.cell
def _(discord_idx, m, motif_idx, mp, sales_ts, selected_item, selected_store):
    # --- Visualization Cell ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Raw Sales Time Series
    ax1.plot(sales_ts, color="#2b5c8f", lw=1.2, label="Daily Unit Sales")
    if motif_idx is not None:
        ax1.axvspan(motif_idx, motif_idx + m, color="green", alpha=0.3, label="Top Motif")
    if discord_idx is not None:
        ax1.axvspan(discord_idx, discord_idx + m, color="red", alpha=0.3, label="Top Discord (Anomaly)")
    ax1.set_title(f"M5 Sales & STUMPY Insights - {selected_item} @ {selected_store}")
    ax1.set_ylabel("Sales Count")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Matrix Profile Distance Trace
    if len(mp) > 0:
        ax2.plot(mp, color="#d95f02", lw=1.2, label=f"Matrix Profile (m={m})")
        ax2.set_ylabel("Matrix Profile Distance")
        ax2.set_xlabel("Time (Days)")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Automated Batch STUMPY Pipeline for Marimo
    """)
    return


@app.cell
def _():
    mo.md(r"""
    # M5 Automated Dataset-Wide Anomaly & Discord Miner
    Scanning all item-store combinations in parallel using STUMPY matrix profiles.
    """)
    return


@app.cell
def _():
    # Configuration controls for batch processing
    limit_slider = mo.ui.slider(start=100, stop=30490, step=500, value=1000, label="Number of Series to Scan (Limit)")
    window_slider_batch = mo.ui.slider(start=7, stop=30, step=1, value=28, label="Matrix Profile Window (m)")

    mo.hstack([limit_slider, window_slider_batch])
    return (limit_slider,)


@app.cell
def _(day_cols, limit_slider, sales_df, window_slider):
    # --- Worker Function for Parallel Processing ---
    def compute_series_anomaly(row, day_cols, m):
        ts = row[day_cols].values.astype(np.float64)
    
        # Skip flat or zero-sales series to avoid division by zero / NaN in matrix profile
        if np.all(ts == 0) or np.std(ts) == 0:
            return None
        
        try:
            # Compute matrix profile
            mp_result = stumpy.stump(ts, m=m)
            mp = mp_result[:, 0]
        
            # The top discord is the maximum distance in the matrix profile
            discord_idx = int(np.argmax(mp))
            max_distance = float(mp[discord_idx])
        
            return {
                "item_id": row["item_id"],
                "store_id": row["store_id"],
                "cat_id": row["cat_id"],
                "dept_id": row["dept_id"],
                "max_mp_distance": max_distance,
                "discord_day_index": discord_idx,
            }
        except Exception:
            return None

    # Slice dataset for execution (use entire sales_df if compute power allows)
    subset_df = sales_df.head(limit_slider.value)
    m_val = window_slider.value

    # Run parallel batch processing across CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(compute_series_anomaly)(row, day_cols, m_val) 
        for _, row in subset_df.iterrows()
    )

    # Filter out None values and build master anomaly DataFrame
    anomaly_records = [r for r in results if r is not None]
    anomaly_df = pd.DataFrame(anomaly_records)

    if not anomaly_df.empty:
        anomaly_df = anomaly_df.sort_values(by="max_mp_distance", ascending=False).reset_index(drop=True)
    return (anomaly_df,)


@app.cell
def _(anomaly_df):
    # Display top global anomalies across the dataset
    mo.md(
        f"""
        ### Top Global Anomalies Discovered ({len(anomaly_df)} series analyzed)
        The items below exhibit the highest matrix profile distances, representing extreme structural breaks, sudden inventory stockouts, or unprecedented demand spikes.
        """
    )
    return


@app.cell
def _(anomaly_df):
    # Render interactive table of top anomalies
    mo.ui.table(round(anomaly_df.head(20), 2), pagination=True)
    return


@app.cell
def _():
    # Display top global anomalies across the dataset
    mo.md(f"""
    ### Select the optimal window size based on Fast Fourier Transform (FFT)
    """)
    return


@app.function
def determine_optimal_window(ts, min_m: int = 7, max_m: int = 60) -> int:
  """Determines the dominant periodicity of a time series using FFT

  restricted strictly within the [min_m, max_m] window range.
  """
  if isinstance(ts, (pd.Series, pd.DataFrame)):
    day_cols = [c for c in ts.index if str(c).startswith("d_")]
    values = (
        ts[day_cols].values if day_cols else pd.to_numeric(ts).values
    )
  else:
    values = np.asarray(ts)

  ts_clean = np.nan_to_num(values.astype(np.float64))
  if np.std(ts_clean) == 0 or len(ts_clean) < max_m * 2:
    return min_m

  # Remove mean (DC component)
  demeaned = ts_clean - np.mean(ts_clean)
  fft_vals = np.abs(np.fft.rfft(demeaned))
  frequencies = np.fft.rfftfreq(len(demeaned))
  fft_vals[0] = 0  # Ignore zero frequency

  # Convert min_m and max_m to frequency bounds (f = 1 / m)
  min_freq = 1.0 / max_m
  max_freq = 1.0 / min_m

  # Restrict search strictly to the valid window bounds
  valid_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
  if not np.any(valid_mask):
    return 28

  masked_fft = np.where(valid_mask, fft_vals, 0)
  peak_idx = np.argmax(masked_fft)
  dominant_freq = frequencies[peak_idx]

  if dominant_freq > 0:
    estimated_m = int(round(1.0 / dominant_freq))
    return max(min_m, min(estimated_m, max_m))

  return 28


@app.function
def analyze_dataset_window_distribution(
    sales_df, day_cols, sample_size=2000, min_m=7, max_m=60
):
  # Sample a representative subset of items for speed, or use len(sales_df) for all 30k
  subset = sales_df.sample(
      n=min(sample_size, len(sales_df)), random_state=42
  )

  print(
      f"Computing optimal window sizes across {len(subset)} M5 time series..."
  )

  # Parallel execution across CPU cores
  results = Parallel(n_jobs=-1)(
      delayed(determine_optimal_window)(row[day_cols].values, min_m, max_m)
      for _, row in subset.iterrows()
  )

  # Compile distribution frequency
  dist_series = pd.Series(results).value_counts().sort_index()
  dist_df = pd.DataFrame(
      {"window_size_m": dist_series.index, "count": dist_series.values}
  )
  dist_df["percentage"] = (
      dist_df["count"] / dist_df["count"].sum()
  ) * 100

  return dist_df


@app.cell
def _(day_cols, sales_df):
    window_dist_df = analyze_dataset_window_distribution(sales_df, day_cols, sample_size=5000, min_m=7, max_m=60)
    window_dist_df.round(2)
    return


@app.cell
def _():
    mo.md(f"""
    ### Time series chain to track pattern drift
    """)
    return


@app.function
def extract_time_series_chain(ts: np.ndarray, m: int):
  """Computes the matrix profile with directional indices and extracts

  the longest unanchored time series chain to track pattern drift.
  """
  ts_clean = np.nan_to_num(ts.astype(np.float64))
  if len(ts_clean) < m * 2:
    return None, []

  # stumpy.stump returns 4 columns:
  # [0]: Distance, [1]: Index, [2]: Left Index (IL), [3]: Right Index (IR)
  mp_result = stumpy.stump(ts_clean, m=m)

  left_indices = mp_result[:, 2].astype(np.int64)
  right_indices = mp_result[:, 3].astype(np.int64)

  # Extract all chains and the longest unanchored chain
  all_chains, longest_chain = chains.allc(left_indices, right_indices)

  return longest_chain, all_chains


@app.cell
def _(day_cols, sales_df):

    m_opt = 7 # or use your dynamic window from Step 1
    sample_ts = sales_df.iloc[0][day_cols].values.astype(np.float64)
    longest_chain_indices, all_chains_list = extract_time_series_chain(sample_ts, m=m_opt)

    print(f"Longest chain sequence indices: {longest_chain_indices}")
    print(f"Total distinct chains found: {len(all_chains_list)}")
    return


@app.function
def extract_regime_shifts(ts: np.ndarray, m: int, n_regimes: int = 2):
  """Computes the matrix profile, runs FLUSS semantic segmentation,

  and returns the Corrected Arc Curve (CAC) and detected regime change indices.
  """
  ts_clean = np.nan_to_num(ts.astype(np.float64))
  if len(ts_clean) < m * 4:
    return None, [], []

  # 1. Compute Matrix Profile
  mp_result = stumpy.stump(ts_clean, m=m)
  mpi = mp_result[:, 1].astype(np.int64)

  # 2. Run FLUSS to find Corrected Arc Curve (CAC) and change points
  # L is typically set equal to the window size m
  cac, regime_locs = stumpy.fluss(mpi, L=m, n_regimes=n_regimes, excl_factor=1)

  return cac, regime_locs, mp_result


@app.cell
def _(day_cols, sales_df):
    def _():
        # --- Example usage on a single M5 sales series ---
        m_opt = 28
        sample_ts = sales_df.iloc[0][day_cols].values.astype(np.float64)
        cac_curve, change_points, mp_res = extract_regime_shifts(sample_ts, m=m_opt, n_regimes=2)
        return print(f"Detected regime change indices (days): {change_points}")


    _()
    return


@app.cell
def _():
    def calculate_adi_cv2(ts: np.ndarray):
      ts_clean = np.nan_to_num(ts.astype(np.float64))
      total_periods = len(ts_clean)

      non_zero_mask = ts_clean > 0
      num_non_zero = np.sum(non_zero_mask)

      if num_non_zero == 0:
        return np.nan, np.nan

      # ADI: Average Demand Interval
      adi = total_periods / num_non_zero

      # CV^2: Squared Coefficient of Variation of non-zero values
      non_zero_vals = ts_clean[non_zero_mask]
      if num_non_zero == 1:
        cv_sq = 0.0
      else:
        mean_val = np.mean(non_zero_vals)
        if mean_val == 0:
          cv_sq = 0.0
        else:
          std_val = np.std(non_zero_vals, ddof=1)
          cv_sq = (std_val / mean_val) ** 2

      return adi, cv_sq


    def classify_sbc(adi, cv_sq):
      if np.isnan(adi) or np.isnan(cv_sq):
        return "Zero/Invalid"

      is_intermittent = adi >= 1.32
      is_erratic = cv_sq >= 0.49

      if not is_intermittent and not is_erratic:
        return "Smooth"
      elif not is_intermittent and is_erratic:
        return "Erratic"
      elif is_intermittent and not is_erratic:
        return "Intermittent"
      else:
        return "Lumpy"


    def run_dataset_sbc_classification(sales_df, day_cols):
      print(
          "Classifying all time series into Syntetos-Boylan-Croston (SBC) types..."
      )

      def worker(row):
        ts = row[day_cols].values
        adi, cv_sq = calculate_adi_cv2(ts)
        category = classify_sbc(adi, cv_sq)
        return {
            "item_id": row["item_id"],
            "store_id": row["store_id"],
            "cat_id": row["cat_id"],
            "adi": adi,
            "cv2": cv_sq,
            "sbc_category": category,
        }

      # Parallel execution across all CPU cores for speed
      results = Parallel(n_jobs=-1)(
          delayed(worker)(row) for _, row in sales_df.iterrows()
      )
      return pd.DataFrame(results)

    return (run_dataset_sbc_classification,)


@app.cell
def _(day_cols, run_dataset_sbc_classification, sales_df):
    sbc_results_df = run_dataset_sbc_classification(sales_df, day_cols)
    print(sbc_results_df['sbc_category'].value_counts())
    return (sbc_results_df,)


@app.function
def plot_sbc_distribution(sbc_df):
  plt.figure(figsize=(10, 6))
  order = ["Smooth", "Erratic", "Intermittent", "Lumpy"]

  # Plot grouped bar chart by category
  sns.countplot(
      data=sbc_df, x="sbc_category", orient="h", order=order, palette="Set2"
  )

  plt.title(
      "M5 Dataset Demand Classification (Syntetos-Boylan-Croston Matrix)",
      fontsize=14,
      pad=15,
  )
  plt.xlabel("SBC Demand Category", fontsize=12)
  plt.ylabel("Number of Item-Store Time Series", fontsize=12)
  plt.legend(title="M5 Category")
  plt.grid(True, axis="y", linestyle="--", alpha=0.5)
  plt.tight_layout()
  plt.show()


@app.cell
def _(sbc_results_df):
    plot_sbc_distribution(sbc_results_df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
