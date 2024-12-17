import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import seaborn as sns

# Function to extract parameters from the dataset
def extract_parameters(csv_file):
    """
    Reads the CSV file and computes required parameters:
    m0, n0, v0, s0 (from prior data)
    Y_topbar, n, v, s (from observed data)
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Split into prior and observed data
    prior_data = data['prior_data'].dropna()
    observed_data = data['observed_data'].dropna()

    # Calculate prior parameters
    m0 = prior_data.mean()
    n0 = len(prior_data)
    v0 = n0 - 1
    s0 = prior_data.std()

    # Calculate observed parameters
    Y_topbar = observed_data.mean()
    n = len(observed_data)
    v = n - 1
    s = observed_data.std()

    return m0, n0, v0, s0, Y_topbar, n, v, s

# Function to perform Bayesian simulation
def bayesian_simulation(m0, n0, v0, s0, Y_topbar, n, v, s):
    """
    Perform Bayesian simulation and generate prior and posterior distributions.
    """
    # Compute posterior parameters
    mn = (Y_topbar * n + n0 * m0) / (n + n0)
    nn = n0 + n
    vn = v0 + n
    s2_n = 1 / vn * (s**2 * (n - 1) + s0**2 * v0 + n0 * n/nn * (Y_topbar - m0)**2)
    sn = np.sqrt(s2_n)

    # Monte Carlo sample size
    sample_size = 50000

    # Prior distributions
    prior_tau_samples = gamma.rvs(a=v0 / 2, scale=2 / (v0 * s0**2), size=sample_size)
    prior_sigma_samples = np.sqrt(1 / prior_tau_samples)
    prior_mu_samples = norm.rvs(loc=m0, scale=prior_sigma_samples / np.sqrt(n0), size=sample_size)

    # Posterior distributions
    tau_samples = gamma.rvs(a=vn / 2, scale=2 / (vn * sn**2), size=sample_size)
    sigma_samples = np.sqrt(1 / tau_samples)
    mu_samples = norm.rvs(loc=mn, scale=sigma_samples / np.sqrt(nn), size=sample_size)

    # Prior predictive distribution
    prior_predictive_samples = norm.rvs(loc=prior_mu_samples, scale=prior_sigma_samples, size=sample_size)

    # Posterior predictive distribution
    posterior_predictive_samples = norm.rvs(loc=mu_samples, scale=sigma_samples, size=sample_size)

    # Calculate 95% credible intervals
    prior_mu_interval = np.percentile(prior_mu_samples, [2.5, 97.5])
    prior_sigma_interval = np.percentile(prior_sigma_samples, [2.5, 97.5])
    posterior_mu_interval = np.percentile(mu_samples, [2.5, 97.5])
    posterior_sigma_interval = np.percentile(sigma_samples, [2.5, 97.5])

    # Print 95% credible intervals
    print(f"Prior 95% credible interval for μ: {prior_mu_interval}")
    print(f"Prior 95% credible interval for σ: {prior_sigma_interval}")
    print(f"Posterior 95% credible interval for μ: {posterior_mu_interval}")
    print(f"Posterior 95% credible interval for σ: {posterior_sigma_interval}")

    # Plotting

    # Prior μ
    plt.figure(figsize=(10, 6))
    plt.hist(prior_mu_samples, bins=20, density=True, color='lightblue', edgecolor='black', alpha=0.7, label='Prior μ')
    plt.axvline(prior_mu_interval[0], color='red', linestyle='--', label='95% CI')
    plt.axvline(prior_mu_interval[1], color='red', linestyle='--')
    plt.xlabel('Prior Mean (μ)', fontsize=28)
    plt.ylabel('Probability Density', fontsize=28)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('Prior Distribution of μ', fontsize=24)

    # Posterior μ
    plt.figure(figsize=(10, 6))
    plt.hist(mu_samples, bins=20, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Posterior μ')
    plt.axvline(posterior_mu_interval[0], color='red', linestyle='--', label='95% CI')
    plt.axvline(posterior_mu_interval[1], color='red', linestyle='--')
    plt.xlabel('Posterior Mean (μ)', fontsize=28)
    plt.ylabel('Probability Density', fontsize=28)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('Posterior Distribution of μ', fontsize=24)

    # Prior σ
    plt.figure(figsize=(10, 6))
    plt.hist(prior_sigma_samples, bins=20, density=True, color='lightgreen', edgecolor='black', alpha=0.7, label='Prior σ')
    plt.axvline(prior_sigma_interval[0], color='red', linestyle='--', label='95% CI')
    plt.axvline(prior_sigma_interval[1], color='red', linestyle='--')
    plt.xlabel('Prior Standard Deviation (σ)', fontsize=28)
    plt.ylabel('Probability Density', fontsize=28)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('Prior Distribution of σ', fontsize=24)

    # Posterior σ
    plt.figure(figsize=(10, 6))
    plt.hist(sigma_samples, bins=20, density=True, color='lightgreen', edgecolor='black', alpha=0.7, label='Posterior σ')
    plt.axvline(posterior_sigma_interval[0], color='red', linestyle='--', label='95% CI')
    plt.axvline(posterior_sigma_interval[1], color='red', linestyle='--')
    plt.xlabel('Posterior Standard Deviation (σ)', fontsize=28)
    plt.ylabel('Probability Density', fontsize=28)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('Posterior Distribution of σ', fontsize=24)

    # Joint Prior Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(x=prior_mu_samples, y=prior_sigma_samples, bins=30, cmap='Reds', cbar=True)
    plt.axvline(prior_mu_interval[0], color='red', linestyle='--', label='95% CI (μ)')
    plt.axvline(prior_mu_interval[1], color='red', linestyle='--')
    plt.axhline(prior_sigma_interval[0], color='green', linestyle='--', label='95% CI (σ)')
    plt.axhline(prior_sigma_interval[1], color='green', linestyle='--')
    plt.xlabel('Prior Mean (μ)', fontsize=28)
    plt.ylabel('Prior Standard Deviation (σ)', fontsize=28)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)    
    plt.title('Joint Prior Distribution of μ and σ', fontsize=24)

    # Joint Posterior Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(x=mu_samples, y=sigma_samples, bins=30, cmap='Reds', cbar=True)
    plt.axvline(posterior_mu_interval[0], color='red', linestyle='--', label='95% CI (μ)')
    plt.axvline(posterior_mu_interval[1], color='red', linestyle='--')
    plt.axhline(posterior_sigma_interval[0], color='green', linestyle='--', label='95% CI (σ)')
    plt.axhline(posterior_sigma_interval[1], color='green', linestyle='--')
    plt.xlabel('Posterior Mean (μ)', fontsize=28)
    plt.ylabel('Posterior Standard Deviation (σ)', fontsize=28)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.title('Joint Posterior Distribution of μ and σ', fontsize=24)

    # Display all plots simultaneously
    plt.show()

# Main function to execute the workflow
def main():
    # CSV file containing the input dataset
    csv_file = 'input.csv'

    # Extract parameters from the CSV
    m0, n0, v0, s0, Y_topbar, n, v, s = extract_parameters(csv_file)

    # Perform Bayesian simulation
    bayesian_simulation(m0, n0, v0, s0, Y_topbar, n, v, s)

if __name__ == "__main__":
    main()
