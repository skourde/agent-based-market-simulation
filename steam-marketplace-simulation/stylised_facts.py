"""
Stylised facts validation module.

Checks whether the simulated market behaves like a real financial market.

The tests compare the simulated price series against common empirical market
patterns from Cont (2001), such as fat-tailed returns, weak return autocorrelation,
volatility clustering, and the link between trading volume and volatility.

Returns are pooled across multiple random seeds for distributional tests.
Autocorrelations are computed within each run first, then averaged across runs,
so independent simulation paths are not artificially stitched together.
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from simulation import Simulation

#return and correlation helpers
#convert price series to log returns: r_t = ln(P_t / P_{t-1})
def log_returns(prices):
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))

#sample autocorrelation of series x at lag k
def autocorrelation(x, lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)

    if lag >= n:
        return np.nan

    #compare the series with a lagged version of itself
    num = np.sum(x[:n - lag] * x[lag:])
    den = np.sum(x ** 2)

    return num / den if den > 0 else 0.0

#compute autocorrelation within each run, then average across runs
#avoids creating fake correlations by concatenating independent simulations
def mean_acf_across_runs(series_per_run, lags):
    acfs = []

    for lag in range(1, lags + 1):
        lag_values = []

        for series in series_per_run:
            if len(series) > lag:
                lag_values.append(autocorrelation(series, lag))

        acfs.append(float(np.mean(lag_values)) if lag_values else np.nan)

    return acfs

#Ljung-Box Q test for autocorrelation up to lag k
#returns the Q statistic and p-value for a supplied ACF list
def ljung_box_from_acfs(acfs, n_obs):
    lags = len(acfs)

    #combine autocorrelations across lags into one joint test statistic
    q_stat = n_obs * (n_obs + 2) * sum(
        (acfs[k] ** 2) / (n_obs - (k + 1))
        for k in range(lags)
        if not np.isnan(acfs[k]) and n_obs > (k + 1)
    )

    p = 1 - stats.chi2.cdf(q_stat, df=lags)
    return q_stat, p

#stylised fact tests
#each function checks one empirical property seen in real markets

#1 - measure of linear autocorrelation in returns (should be close to zero)
def sf1_return_autocorrelation(returns_per_run, lags=10):
    acfs = mean_acf_across_runs(returns_per_run, lags)
    max_abs_acf = max(abs(a) for a in acfs if not np.isnan(a))

    #use average run length for the Ljung-Box test because ACFs are averaged per run
    mean_run_length = int(np.mean([len(r) for r in returns_per_run]))
    q, p = ljung_box_from_acfs(acfs, mean_run_length)

    #pass only if correlations are small and the joint autocorrelation test is insignificant
    if max_abs_acf < 0.05 and p > 0.05:
        verdict = "PASS"
    elif max_abs_acf < 0.10:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        'name': 'SF1: Absence of return autocorrelation',
        'metric': f'max |rho(k)| over lags 1-{lags}, averaged across runs',
        'value': max_abs_acf,
        'benchmark': '< 0.05 (Cont 2001)',
        'ljung_box_q': q,
        'ljung_box_p': p,
        'acfs': acfs,
        'verdict': verdict,
    }

#2 - measure of tail heaviness using Pearson kurtosis (normal distribution = 3)
def sf2_heavy_tails(returns_pooled):
    k = stats.kurtosis(returns_pooled, fisher=False)

    #real return distributions usually have fatter tails than a normal distribution
    if 4 <= k <= 50:
        verdict = "PASS"
    elif 3 < k < 4 or 50 < k < 100:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        'name': 'SF2: Heavy tails (excess kurtosis)',
        'metric': 'Pearson kurtosis of returns',
        'value': k,
        'benchmark': '4-50 typical, 3 = Gaussian (Cont 2001)',
        'verdict': verdict,
    }

#3 - measure of autocorrelation in absolute returns (volatility clustering)
def sf3_volatility_clustering(returns_per_run, lags=20):
    abs_returns_per_run = [np.abs(r) for r in returns_per_run]

    #absolute returns proxy volatility, so autocorrelation here detects clustering
    acfs = mean_acf_across_runs(abs_returns_per_run, lags)
    mean_acf = float(np.nanmean(acfs))
    first_acf = acfs[0]

    #pass if volatility is clearly persistent across several lags
    if mean_acf > 0.05 and first_acf > 0.10:
        verdict = "PASS"
    elif mean_acf > 0.02:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        'name': 'SF3: Volatility clustering',
        'metric': f'mean rho(|r|, k) over lags 1-{lags}, averaged across runs',
        'value': mean_acf,
        'first_lag_acf': first_acf,
        'acfs': acfs,
        'benchmark': '0.10-0.30, slow decay (Cont 2001)',
        'verdict': verdict,
    }

#4 - measure of how kurtosis changes as returns are aggregated over longer windows
def sf4_aggregational_gaussianity(returns_per_run, aggregation_levels=(1, 5, 10, 20)):
    results = []

    for k_agg in aggregation_levels:
        aggregated = []

        for run_returns in returns_per_run:
            r = np.asarray(run_returns)

            #split returns into non-overlapping windows of size k_agg
            n_blocks = len(r) // k_agg

            if n_blocks < 2:
                continue

            #aggregate returns within each window to simulate lower-frequency returns
            blocks = r[:n_blocks * k_agg].reshape(n_blocks, k_agg).sum(axis=1)

            #pool aggregated returns across all runs so kurtosis is measured on a large sample
            aggregated.extend(blocks.tolist())

        if len(aggregated) > 5:
            k = stats.kurtosis(aggregated, fisher=False)
            results.append((k_agg, k, len(aggregated)))

    #kurtosis should fall at every aggregation step, not just from first to last
    kurts = [r[1] for r in results]

    if len(kurts) >= 2:
        strictly_decreasing = all(
            kurts[i] >= kurts[i + 1]
            for i in range(len(kurts) - 1)
        )

        if strictly_decreasing and kurts[-1] < kurts[0] - 0.5:
            verdict = "PASS"
        elif strictly_decreasing:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"
    else:
        verdict = "INSUFFICIENT DATA"

    return {
        'name': 'SF4: Aggregational Gaussianity',
        'metric': 'kurtosis vs aggregation level',
        'value': results,
        'benchmark': 'monotonic decay toward 3 (Cont 2001)',
        'verdict': verdict,
    }

#5 - measure of correlation between trading volume and absolute returns (volatility)
def sf5_volume_volatility(runs_data):
    correlations = []

    for run in runs_data:
        prices = run['prices']
        volumes = run['volume']

        rets = log_returns(prices)
        abs_rets = np.abs(rets)

        #align volume with returns because returns are one element shorter than prices
        vol = np.asarray(volumes[:len(abs_rets)], dtype=float)

        #skip flat series because correlation is undefined if either side has no variation
        if vol.std() > 0 and abs_rets.std() > 0:
            r, _ = stats.pearsonr(vol, abs_rets)

            #clip avoids infinite Fisher z values if correlation is exactly ±1
            correlations.append(np.clip(r, -0.999999, 0.999999))

    if correlations:
        #Fisher z-transform makes averaging correlations less biased
        z_values = np.arctanh(correlations)
        mean_z = float(np.mean(z_values))
        mean_corr = float(np.tanh(mean_z))

        if len(z_values) > 1:
            se_z = float(np.std(z_values, ddof=1) / math.sqrt(len(z_values)))
            z_low = mean_z - 1.96 * se_z
            z_high = mean_z + 1.96 * se_z

            #convert the confidence interval back from z-space to correlation space
            ci_low = float(np.tanh(z_low))
            ci_high = float(np.tanh(z_high))
            ci = max(mean_corr - ci_low, ci_high - mean_corr)
        else:
            ci = 0.0
    else:
        mean_corr = 0.0
        ci = 0.0

    if 0.15 <= mean_corr <= 0.6:
        verdict = "PASS"
    elif 0.05 <= mean_corr < 0.15:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        'name': 'SF5: Volume-volatility correlation',
        'metric': 'mean Pearson rho(volume, |return|) across runs',
        'value': mean_corr,
        'ci_95': ci,
        'benchmark': '0.20-0.50 (Karpoff 1987)',
        'verdict': verdict,
    }

#run multiple simulations with different seeds, pool returns, and evaluate stylised facts
def run_validation(config, n_seeds=30, label='Baseline'):
    runs_data = []
    returns_per_run = []

    for seed in range(n_seeds):
        #reset random seed so each run is independent but reproducible
        random.seed(seed)

        sim = Simulation(config)
        prices = sim.run()

        #store prices and trading volume so both return-based and volume-based tests can be run
        runs_data.append({'prices': prices, 'volume': sim.market.trades_executed})
        returns_per_run.append(log_returns(prices))

    #pool returns from all seeds into one large sample for distributional tests
    pooled = np.concatenate(returns_per_run)

    results = [
        sf1_return_autocorrelation(returns_per_run),
        sf2_heavy_tails(pooled),
        sf3_volatility_clustering(returns_per_run),
        sf4_aggregational_gaussianity(returns_per_run),
        sf5_volume_volatility(runs_data),
    ]

    return {
        'label': label,
        'n_seeds': n_seeds,
        'n_returns_pooled': len(pooled),
        'pooled_returns': pooled,
        'returns_per_run': returns_per_run,
        'results': results,
    }

#printing helper for stylised facts validation results
def print_report(validation):
    print(f"\n{'=' * 72}")
    print(f"Stylised Facts Validation: {validation['label']}")
    print(f"({validation['n_seeds']} seeds, {validation['n_returns_pooled']} returns pooled)")
    print(f"Benchmark: Cont (2001), Quantitative Finance 1(2)")
    print(f"{'=' * 72}")

    for r in validation['results']:
        print(f"\n{r['name']}")
        print(f"  Metric:    {r['metric']}")

        if isinstance(r['value'], list):
            for k, kurt, n in r['value']:
                print(f"             aggregation={k:>2}  kurtosis={kurt:7.3f}  (n={n})")
        else:
            extra = f" ± {r['ci_95']:.3f}" if 'ci_95' in r else ''
            print(f"  Value:     {r['value']:.4f}{extra}")

        print(f"  Benchmark: {r['benchmark']}")
        print(f"  Verdict:   {r['verdict']}")

        if 'ljung_box_p' in r:
            print(f"  Ljung-Box p-value: {r['ljung_box_p']:.4f}")

#diagnostic plots for visualising stylised facts validation results
def plot_diagnostics(validation, save_path='stylised_facts_diagnostics.png'):
    pooled = validation['pooled_returns']
    returns_per_run = validation['returns_per_run']

    #map SF1/SF2/etc. to their result dictionaries for easier plot access
    results = {r['name'].split(':')[0]: r for r in validation['results']}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    #use per-run length for the confidence band because ACFs are averaged across runs
    mean_run_length = np.mean([len(r) for r in returns_per_run])
    ci_band = 1.96 / math.sqrt(mean_run_length)

    #top left: SF1 - ACF of raw returns
    ax = axes[0, 0]
    acfs = results['SF1']['acfs']

    ax.bar(range(1, len(acfs) + 1), acfs, color='steelblue', alpha=0.8)
    ax.axhline(ci_band, color='r', linestyle='--', linewidth=0.8, label='95% CI')
    ax.axhline(-ci_band, color='r', linestyle='--', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.5)

    ax.set_title('SF1: No Linear Autocorrelation\nACF of Returns, Avg Across Runs')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    #top right: SF2 - return distribution vs Gaussian
    ax = axes[0, 1]

    ax.hist(pooled, bins=80, density=True, alpha=0.6,
            color='steelblue', label='Simulated')

    mu, sigma = pooled.mean(), pooled.std()
    xs = np.linspace(pooled.min(), pooled.max(), 200)

    #plot a normal distribution with same mean/std for comparison
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma),
            'r--', linewidth=1.5,
            label=f'Gaussian fit (σ={sigma:.3f})')

    ax.set_yscale('log')

    ax.set_title('SF2: Heavy Tails\nReturn Distribution (log scale)')
    ax.set_xlabel('Log return')
    ax.set_ylabel('Density (log)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    #bottom left: SF3 - ACF of absolute returns
    ax = axes[1, 0]
    acfs_abs = results['SF3']['acfs']

    #positive bars indicate volatility clustering
    ax.bar(range(1, len(acfs_abs) + 1),
        acfs_abs,
        color='darkorange',
        alpha=0.8)

    ax.axhline(ci_band, color='r', linestyle='--',
            linewidth=0.8, label='95% CI')
    ax.axhline(-ci_band, color='r', linestyle='--', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.5)

    ax.set_title('SF3: Volatility Clustering\nACF of |Returns|, Avg Across Runs')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    #bottom right: SF4 - kurtosis vs aggregation
    ax = axes[1, 1]
    agg_data = results['SF4']['value']

    if agg_data:
        ks = [r[0] for r in agg_data]
        kurts = [r[1] for r in agg_data]

        #kurtosis should decline toward 3 as aggregation increases
        ax.plot(ks, kurts,
                'o-',
                color='steelblue',
                linewidth=1.5,
                markersize=8,
                label='Simulated kurtosis')

        ax.axhline(3,
                color='r',
                linestyle='--',
                linewidth=1.0,
                label='Gaussian (k=3)')

        ax.set_title('SF4: Aggregational Gaussianity\nKurtosis vs Aggregation Level')
        ax.set_xlabel('Aggregation window (ticks)')
        ax.set_ylabel('Kurtosis')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Stylised Facts Validation: {validation['label']}",
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

#main
if __name__ == '__main__':
    #validate the baseline mixed regime — most stable, most representative
    BASELINE = {
        'casual': 8,
        'collectors': 8,
        'speculators': 4,
        'steps': 200,
    }

    print("Running stylised facts validation on baseline configuration...")
    validation = run_validation(BASELINE, n_seeds=30, label='Baseline (Mixed)')
    print_report(validation)
    plot_diagnostics(validation, save_path='stylised_facts_diagnostics.png')