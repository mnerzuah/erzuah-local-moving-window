"""
erzuah-local-moving-window
Python script for cleaning Measurement While Drilling (MWD) data
using a local moving window outlier detection approach.

Note
This implementation was developed with assistance from AI tools and has been reviewed, validated, and adapted by the author.

HOW IT WORKS
------------
For each row, the script:
1. Looks at neighboring rows before and after it
2. Computes a local mean from the neighbors
3. Flags outliers based on user-defined thresholds:
   - Penetration rate below 80% of local mean
   - Thrust above 120% of local mean
4. Removes flagged rows
5. Saves both cleaned data and flagged rows

EDIT THE PLACEHOLDERS BELOW BEFORE RUNNING.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# USER INPUTS — EDIT THESE
# =========================================================

# Input Excel or CSV file
INPUT_FILE = r"PATH_TO_YOUR_INPUT_FILE.xlsx"   # <-- replace with your input file path

# Sheet name if using Excel
SHEET_NAME = "Sheet1"                          # <-- replace if needed

# Depth column
DEPTH_COL = "Prof [DEPTH][m]"                  # <-- replace with your depth column name

# Thrust column
THRUST_COL = "FO [TPAF][lbf]"                  # <-- replace with your thrust column name

# Penetration rate column
PEN_RATE_COL = "VA [AS][ft/min]"               # <-- replace with your penetration rate column name

# Moving window size:
# window_radius = 3 means 3 rows before + 3 rows after
WINDOW_RADIUS = 3

# Thresholds
PEN_RATE_LOWER_RATIO = 0.80   # remove if penetration rate < 80% of local mean
THRUST_UPPER_RATIO = 1.20     # remove if thrust > 120% of local mean

# Output folder
OUTPUT_FOLDER = r"PATH_TO_OUTPUT_FOLDER"       # <-- replace with output folder path


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_data(file_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """
    Load CSV or Excel file into a DataFrame.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, sheet_name=sheet_name)

    raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """
    Check that required columns exist in the DataFrame.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def local_mean(series: pd.Series, center_idx: int, radius: int) -> float:
    """
    Compute local mean using neighboring values only,
    excluding the center point itself.
    """
    start_idx = max(0, center_idx - radius)
    end_idx = min(len(series), center_idx + radius + 1)

    window = series.iloc[start_idx:end_idx].copy()

    # Drop the center value
    center_label = series.index[center_idx]
    window = window.drop(center_label, errors="ignore")

    # Convert to numeric and remove NaN
    window = pd.to_numeric(window, errors="coerce").dropna()

    if len(window) == 0:
        return np.nan

    return window.mean()


def apply_local_moving_window_cleaning(
    df: pd.DataFrame,
    thrust_col: str,
    pen_rate_col: str,
    window_radius: int,
    pen_rate_lower_ratio: float,
    thrust_upper_ratio: float
) -> pd.DataFrame:
    """
    Apply Erzuah's Local Moving Window Method.
    Flags rows for removal if:
      - penetration rate < pen_rate_lower_ratio * local mean
      - thrust > thrust_upper_ratio * local mean
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    df[thrust_col] = pd.to_numeric(df[thrust_col], errors="coerce")
    df[pen_rate_col] = pd.to_numeric(df[pen_rate_col], errors="coerce")

    # Store local means for transparency/debugging
    df["Local Mean - Penetration Rate"] = np.nan
    df["Local Mean - Thrust"] = np.nan

    # Store flags
    df["Flag Low Penetration Rate"] = False
    df["Flag High Thrust"] = False
    df["Remove Row"] = False

    for i in range(len(df)):
        pen_local = local_mean(df[pen_rate_col], i, window_radius)
        thrust_local = local_mean(df[thrust_col], i, window_radius)

        df.at[df.index[i], "Local Mean - Penetration Rate"] = pen_local
        df.at[df.index[i], "Local Mean - Thrust"] = thrust_local

        current_pen = df.iloc[i][pen_rate_col]
        current_thrust = df.iloc[i][thrust_col]

        # Flag low penetration rate
        if pd.notna(current_pen) and pd.notna(pen_local) and pen_local != 0:
            if current_pen < pen_rate_lower_ratio * pen_local:
                df.at[df.index[i], "Flag Low Penetration Rate"] = True

        # Flag high thrust
        if pd.notna(current_thrust) and pd.notna(thrust_local) and thrust_local != 0:
            if current_thrust > thrust_upper_ratio * thrust_local:
                df.at[df.index[i], "Flag High Thrust"] = True

    # Final removal flag
    df["Remove Row"] = df["Flag Low Penetration Rate"] | df["Flag High Thrust"]

    return df


def save_outputs(
    cleaned_df: pd.DataFrame,
    flagged_df: pd.DataFrame,
    full_df: pd.DataFrame,
    output_folder: str,
    input_file: str
) -> None:
    """
    Save:
    1. cleaned data
    2. flagged/removed rows
    3. full annotated file
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(input_file).stem

    cleaned_file = output_path / f"{base_name}_cleaned.xlsx"
    flagged_file = output_path / f"{base_name}_flagged_rows.xlsx"
    full_file = output_path / f"{base_name}_annotated.xlsx"

    cleaned_df.to_excel(cleaned_file, index=False)
    flagged_df.to_excel(flagged_file, index=False)
    full_df.to_excel(full_file, index=False)

    print(f"Saved cleaned file: {cleaned_file}")
    print(f"Saved flagged rows file: {flagged_file}")
    print(f"Saved annotated file: {full_file}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    # Load data
    df = load_data(INPUT_FILE, SHEET_NAME)

    # Validate required columns
    validate_columns(df, [DEPTH_COL, THRUST_COL, PEN_RATE_COL])

    # Sort by depth if possible
    df[DEPTH_COL] = pd.to_numeric(df[DEPTH_COL], errors="coerce")
    df = df.sort_values(by=DEPTH_COL).reset_index(drop=True)

    # Apply cleaning method
    df_annotated = apply_local_moving_window_cleaning(
        df=df,
        thrust_col=THRUST_COL,
        pen_rate_col=PEN_RATE_COL,
        window_radius=WINDOW_RADIUS,
        pen_rate_lower_ratio=PEN_RATE_LOWER_RATIO,
        thrust_upper_ratio=THRUST_UPPER_RATIO
    )

    # Split outputs
    df_flagged = df_annotated[df_annotated["Remove Row"]].copy()
    df_cleaned = df_annotated[~df_annotated["Remove Row"]].copy()

    print(f"Original rows: {len(df_annotated)}")
    print(f"Removed rows: {len(df_flagged)}")
    print(f"Remaining rows: {len(df_cleaned)}")

    # Save outputs
    save_outputs(
        cleaned_df=df_cleaned,
        flagged_df=df_flagged,
        full_df=df_annotated,
        output_folder=OUTPUT_FOLDER,
        input_file=INPUT_FILE
    )


if __name__ == "__main__":
    main()