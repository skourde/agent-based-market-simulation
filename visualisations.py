"""
Experiments module: runs the three population-mix experiments and produces the summary statistics and plots used in the Results chapter.

Each experiment runs N_SEEDS times with different random seeds.
- Reports the mean trajectory, 95% confidence intervals across runs, and summary statistics (mean price, volatility, peak price) with confidence intervals
"""
import csv
import random
import statistics
import matplotlib.pyplot as plt
from simulation import Simulation

#three population-mix experiments
#all other market parameters are at defaults
experiments = {
    "Baseline (Mixed)":     {'casual': 10, 'collectors': 5,  'speculators': 5,  'steps': 300},
    "Speculator-Dominated": {'casual': 5,  'collectors': 2,  'speculators': 15, 'steps': 300},
    "Collector-Dominated":  {'casual': 5,  'collectors': 15, 'speculators': 2,  'steps': 300},
}

N_SEEDS = 30 #number of runs per experiment

#run every experiment N_SEEDS times and collect the price trajectories
def run_all():
    results = {}
    for name, config in experiments.items():
        runs = []
        for seed in range(N_SEEDS):
            random.seed(seed) #make each run reproducible
            sim = Simulation(config)
            prices = sim.run()
            runs.append(prices)
        results[name] = runs
    return results

#plot mean price trajectory for each experiment with a 95% CI band
def plot_experiments(results, save_path='experiment_results.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, runs) in zip(axes, results.items()):
        n_steps = len(runs[0])
        #for each time step, compute mean and 95% CI across runs
        mean_traj = [statistics.mean([r[t] for r in runs]) for t in range(n_steps)]
        std_traj  = [statistics.stdev([r[t] for r in runs]) for t in range(n_steps)]
        ci_traj   = [1.96 * s / (N_SEEDS ** 0.5) for s in std_traj]
        upper = [m + c for m, c in zip(mean_traj, ci_traj)]
        lower = [m - c for m, c in zip(mean_traj, ci_traj)]

        ax.plot(mean_traj, color='steelblue', linewidth=1.2, label='Mean')
        ax.fill_between(range(n_steps), lower, upper, color='steelblue',
                        alpha=0.2, label='95% CI')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

#helper: 95% confidence interval half-width for a list of values
def ci(vals):
    return 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)

#print summary table to console and save to CSV for the dissertation
def print_summary(results, save_path='experiment_summary.csv'):
    print("\nExperiment Summary (mean ± 95% CI across {} seeds):".format(N_SEEDS))
    rows = [['Experiment', 'Mean Price', 'Mean CI', 'Volatility',
             'Volatility CI', 'Peak Price', 'Peak CI']]
    for name, runs in results.items():
        means = [statistics.mean(r) for r in runs]
        vols  = [statistics.stdev(r) for r in runs]
        peaks = [max(r) for r in runs]
        print(f"\n{name}")
        print(f"  Mean Price: {statistics.mean(means):.3f} ± {ci(means):.3f}")
        print(f"  Volatility: {statistics.mean(vols):.3f} ± {ci(vols):.3f}")
        print(f"  Peak Price: {statistics.mean(peaks):.3f} ± {ci(peaks):.3f}")
        rows.append([name,
                     f"{statistics.mean(means):.3f}", f"{ci(means):.3f}",
                     f"{statistics.mean(vols):.3f}",  f"{ci(vols):.3f}",
                     f"{statistics.mean(peaks):.3f}", f"{ci(peaks):.3f}"])
    with open(save_path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)

if __name__ == '__main__':
    results = run_all()
    plot_experiments(results)
    print_summary(results)