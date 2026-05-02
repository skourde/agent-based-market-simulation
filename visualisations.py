import matplotlib.pyplot as plt
import statistics
import random
from simulation import Simulation

experiments = {
    "Baseline (Mixed)": {'casual': 10, 'collectors': 5,  'speculators': 5,  'steps': 300},
    "Speculator-Dominated": {'casual': 5,  'collectors': 2,  'speculators': 15, 'steps': 300},
    "Collector-Dominated": {'casual': 5,  'collectors': 15, 'speculators': 2,  'steps': 300},
}

N_SEEDS = 30

def run_all():
    results = {}
    for name, config in experiments.items():
        runs = []
        for seed in range(N_SEEDS):
            random.seed(seed)
            sim = Simulation(config)
            prices = sim.run()
            runs.append(prices)
        results[name] = runs
    return results

def plot_experiments(results, save_path='experiment_results.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, runs) in zip(axes, results.items()):
        n_steps = len(runs[0])
        mean_traj = [statistics.mean([r[t] for r in runs]) for t in range(n_steps)]
        ax.plot(mean_traj, color='steelblue', linewidth=1.2)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def print_summary(results):
    print("\nExperiment Summary (mean ± 95% CI across 30 seeds):")
    for name, runs in results.items():
        means = [statistics.mean(r) for r in runs]
        vols  = [statistics.stdev(r) for r in runs]
        peaks = [max(r) for r in runs]

        def ci(vals):
            return 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)

        print(f"\n{name}")
        print(f"Mean Price: {statistics.mean(means):.3f} ± {ci(means):.3f}")
        print(f"Volatility: {statistics.mean(vols):.3f} ± {ci(vols):.3f}")
        print(f"Peak Price: {statistics.mean(peaks):.3f} ± {ci(peaks):.3f}")

if __name__ == '__main__':
    results = run_all()
    plot_experiments(results)
    print_summary(results)