import pandas as pd
import numpy as np

# Load your real data
df = pd.read_csv('dataset/m-blue.csv')  # or use your DataFrame directly

np.random.seed(42)  # For reproducibility

def jitter_series(series, rel_std=0.07, abs_std=100):
    """Jitter a pandas Series by a relative and absolute standard deviation."""
    if pd.api.types.is_numeric_dtype(series):
        noise = np.random.normal(0, rel_std * (series.abs() + abs_std))
        new_series = series + noise
        # No negative values
        new_series = np.clip(new_series, 0, None)
        # If integer, round
        if pd.api.types.is_integer_dtype(series):
            new_series = new_series.round().astype(int)
        return new_series
    else:
        return series

# Jitter all columns except date
jittered_df = df.copy()
for col in df.columns:
    if col != 'date':
        jittered_df[col] = jitter_series(df[col])

# Save to new CSV
jittered_df.to_csv('mock_mmm_data.csv', index=False)
print(jittered_df.head())