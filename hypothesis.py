from scipy.stats import chisquare, ks_2samp
import pandas as pd
from sklearn.metrics import accuracy_score


# Load datasets
expected_df = pd.read_csv('./data/credit_train.csv')
observed_df = pd.read_csv('./data/credit_pred_with_predictions.csv')

# Perform Chi-squared test

observed_freq = observed_df['Y'].value_counts(normalize=True)
expected_freq = expected_df['Y'].value_counts(normalize=True)  

print(observed_freq, expected_freq)

chi_stat, p_val_chi = chisquare(observed_freq, expected_freq)
print(f"Chi Stat: {chi_stat}    P-Val: {p_val_chi}")
if p_val_chi < 0.05:
    print("Chi-squared test failed: Distributions are significantly different.")
else:
    print("Chi-squared test passed")

# Apply Kolmogorov-Smirnov test
ks_stat, p_val_ks = ks_2samp(observed_df['Y'], expected_df['Y'])
print(f"KS Stat: {ks_stat}    P-Val: {p_val_ks}")
if p_val_ks < 0.05:
    print("Kolmogorov-Smirnov test failed: Distributions are significantly different.")
else:
    print("Kolmogorov-Smirnov test passed")


