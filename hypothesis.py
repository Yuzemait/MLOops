from scipy.stats import chisquare, ks_2samp
import pandas as pd
from sklearn.model_selection import train_test_split


# Load datasets
df = pd.read_csv('./data/new_data.csv')

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Perform Chi-squared test

observed_freq = test_df['Y'].value_counts(normalize=True)
expected_freq = train_df['Y'].value_counts(normalize=True)


chi_stat, p_val_chi = chisquare(observed_freq, expected_freq)
print(f"Chi Stat: {chi_stat}    P-Val: {p_val_chi}")
if p_val_chi < 0.05:
    print("Chi-squared test failed: Distributions are significantly different.")
else:
    print("Chi-squared test passed")

# Apply Kolmogorov-Smirnov test
ks_stat, p_val_ks = ks_2samp(train_df['Y'], test_df['Y'])
print(f"KS Stat: {ks_stat}    P-Val: {p_val_ks}")
if p_val_ks < 0.05:
    print("Kolmogorov-Smirnov test failed: Distributions are significantly different.")
else:
    print("Kolmogorov-Smirnov test passed")


