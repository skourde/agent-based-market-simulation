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

#experiment 1: population composition (no transaction fee)
#investigates how the mix of agent types affects price dynamics
composition_experiments = {
    "Baseline (Mixed)": {'casual': 8, 'collectors': 8,  'speculators': 4,  'steps': 200},
    "Speculator-Dominated": {'casual': 5,  'collectors': 2,  'speculators': 15, 'steps': 200},
    "Collector-Dominated": {'casual': 5,  'collectors': 15, 'speculators': 2,  'steps': 200},
}

#experiment 2: transaction fee impact (baseline composition throughout)
#investigates how Steam's 15% platform fee affects price stability and trade volume
fee_experiments = {
    "No Fee (0%)": {'casual': 8,  'collectors': 8,  'speculators': 4,  'steps': 200, 'transaction_fee': 0.0},
    "Steam Fee (15%)": {'casual': 8,  'collectors': 8,  'speculators': 4,  'steps': 200, 'transaction_fee': 0.15},
    "High Fee (30%)": {'casual': 8,  'collectors': 8,  'speculators': 4,  'steps': 200, 'transaction_fee': 0.30},
}

N_SEEDS = 30 #number of runs per experiment

#run every experiment N_SEEDS times and collect the price trajectories
def run_all(experiments):
    results = {}
    for name, config in experiments.items():
        runs = []
        for seed in range(N_SEEDS):
            random.seed(seed)
            sim = Simulation(config)
            prices = sim.run()
            # store both price history and trade volume per tick
            runs.append({
                'prices': prices,
                'volume': sim.market.trades_executed
            })
        results[name] = runs
    return results

#plot mean price trajectory for each experiment with a 95% CI band
def plot_experiments(results, save_path='experiment_results.png', title=''):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, runs) in zip(axes, results.items()):
        n_steps = len(runs[0]['prices'])
        mean_traj = [statistics.mean([r['prices'][t] for r in runs]) for t in range(n_steps)]
        std_traj  = [statistics.stdev([r['prices'][t] for r in runs]) for t in range(n_steps)]
        ci_traj   = [1.96 * s / (N_SEEDS ** 0.5) for s in std_traj]
        upper = [m + c for m, c in zip(mean_traj, ci_traj)]
        lower = [m - c for m, c in zip(mean_traj, ci_traj)]
        ax.plot(mean_traj, color='steelblue', linewidth=1.2, label='Mean')
        ax.fill_between(range(n_steps), lower, upper,
                        color='steelblue', alpha=0.2, label='95% CI')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

#helper: 95% confidence interval half-width for a list of values
def ci(vals):
    return 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)

def print_summary(results, label=''):
    print(f"\n{'='*55}")
    print(f"Experiment Summary: {label}")
    print(f"(mean ± 95% CI across {N_SEEDS} seeds)")
    print(f"{'='*55}")
    for name, runs in results.items():
        means  = [statistics.mean(r['prices']) for r in runs]
        vols   = [statistics.stdev(r['prices']) for r in runs]
        peaks  = [max(r['prices']) for r in runs]
        # average trades per tick across the simulation
        avg_volume = [statistics.mean(r['volume']) for r in runs]
        print(f"\n{name}")
        print(f"  Mean Price:       {statistics.mean(means):.3f} ± {ci(means):.3f}")
        print(f"  Volatility:       {statistics.mean(vols):.3f}  ± {ci(vols):.3f}")
        print(f"  Peak Price:       {statistics.mean(peaks):.3f} ± {ci(peaks):.3f}")
        print(f"  Avg Trades/Tick:  {statistics.mean(avg_volume):.3f} ± {ci(avg_volume):.3f}")

if __name__ == '__main__':
    #run and plot composition experiments
    comp_results = run_all(composition_experiments)
    plot_experiments(comp_results,
                     save_path='results_composition.png',
                     title='Experiment 1: Effect of Agent Composition on Price Dynamics')
    print_summary(comp_results, label='Agent Composition')

    #run and plot fee experiments
    fee_results = run_all(fee_experiments)
    plot_experiments(fee_results,
                     save_path='results_fees.png',
                     title='Experiment 2: Effect of Transaction Fee on Price Dynamics')
    print_summary(fee_results, label='Transaction Fee Impact')