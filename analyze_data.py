import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import downsampler from same folder
import downsampling as ds



def analyze(df):
    print(f"Data shape: {df.shape}")

    # --- Check column types ---
    print("\nColumn type summary:")

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            print(f"  {col}: Datetime")

        elif pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique(dropna=True)

            # categorical numeric if <10 unique values
            if n_unique < 10:
                unique_vals = sorted(series.dropna().unique().tolist())
                print(f"  {col}: Categorical (Numeric codes, {n_unique} unique values)")
                print(f"     ➜ Unique values: {unique_vals}")
            else:
                print(f"  {col}: Numeric")

        else:
            n_unique = series.nunique(dropna=True)
            print(f"  {col}: Classification / Text ({n_unique} unique values)")
            if n_unique < 10:
                unique_vals = sorted(series.dropna().unique().tolist())
                print(f"     ➜ Unique values: {unique_vals}")


 # --- Check missing values ---
    missing = df.isna().sum()
    if missing.any():
        print("\nMissing values:")
        print(missing[missing > 0])
    else:
        print("No missing values.")

# --- Create folder for plots ---
    os.makedirs("plots", exist_ok=True)

   # --- Plot distributions for numeric columns ---
    num_cols = df.select_dtypes("number").columns
    if len(num_cols) == 0:
        print("\nNo numeric columns found. Skipping plots.")
        return

    os.makedirs("plots", exist_ok=True)

    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"plots/{col}_distribution.png")
        plt.close()
    print("\nCreating distribution plots...")
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"plots/{col}_distribution.png")
        plt.close()

    # --- Correlation heatmap (exclude Timestamp) ---
    numeric_for_corr = df.select_dtypes("number")
    if len(numeric_for_corr.columns) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_for_corr.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap (Numeric Columns Only)")
        plt.tight_layout()
        plt.savefig("plots/correlation_heatmap.png")
        plt.close()
        print("Correlation heatmap saved in /plots folder.")
    else:
        print("Not enough numeric columns for correlation heatmap.")
 
                
# --- MAIN EXECUTION --- to get hourly data:
# List of streets: Kotaniementie, Saaritie, Kaakkovuorentie, Tähtiniementie, Tuulimyllyntie.
if __name__ == '__main__':
#    # Downsample directly
    df_out = ds.get_hourly('Kaakkovuorentie', lht_included=True)
    print(df_out.head())

    # Run the analysis
    analyze(df_out)

#--- MAIN EXECUTION ---to get ten minutely data:
#if __name__ == '__main__':
#    df_out = ds.get_ten_minutely('Saaritie', lht_included=True)
#    print(df_out.head())

#    analyze(df_out)