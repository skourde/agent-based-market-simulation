"""
Interactive GUI for exploring transaction fee effects on market dynamics.
Adjust the fee slider to re-run the simulation and update the plots live.
"""
import random
import statistics
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from simulation import Simulation

N_SEEDS = 20
BASE_CONFIG = {'casual': 8, 'collectors': 8, 'speculators': 4, 'steps': 200}


def run_experiment(fee):
    config = {**BASE_CONFIG, 'transaction_fee': fee}
    runs = []
    for seed in range(N_SEEDS):
        random.seed(seed)
        sim = Simulation(config)
        prices = sim.run()
        runs.append({'prices': prices, 'volume': sim.market.trades_executed})
    return runs


def compute_trajectories(runs):
    n_steps = len(runs[0]['prices'])
    mean_p = [statistics.mean(r['prices'][t] for r in runs) for t in range(n_steps)]
    ci_p   = [1.96 * statistics.stdev(r['prices'][t] for r in runs) / N_SEEDS**0.5
              for t in range(n_steps)]
    n_ticks = len(runs[0]['volume'])
    mean_v = [statistics.mean(r['volume'][t] for r in runs) for t in range(n_ticks)]
    return mean_p, ci_p, mean_v


def summary_text(runs):
    means  = [statistics.mean(r['prices']) for r in runs]
    vols   = [statistics.stdev(r['prices']) for r in runs]
    avg_v  = [statistics.mean(r['volume']) for r in runs]
    ci = lambda vals: 1.96 * statistics.stdev(vals) / len(vals)**0.5
    return (
        f"Mean Price:      {statistics.mean(means):.3f} ± {ci(means):.3f}\n"
        f"Volatility:      {statistics.mean(vols):.3f} ± {ci(vols):.3f}\n"
        f"Avg Trades/Tick: {statistics.mean(avg_v):.3f} ± {ci(avg_v):.3f}"
    )


# ── initial draw ────────────────────────────────────────────────────────────
INITIAL_FEE = 0.15

fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(8, 7))
fig.subplots_adjust(bottom=0.22, hspace=0.45)

runs = run_experiment(INITIAL_FEE)
mean_p, ci_p, mean_v = compute_trajectories(runs)
steps = list(range(len(mean_p)))
upper = [m + c for m, c in zip(mean_p, ci_p)]
lower = [m - c for m, c in zip(mean_p, ci_p)]

line_p, = ax_price.plot(steps, mean_p, color='steelblue', linewidth=1.5, label='Mean')
fill_ref = [ax_price.fill_between(steps, lower, upper, color='steelblue', alpha=0.2, label='95% CI')]
ax_price.set_xlabel('Time Step')
ax_price.set_ylabel('Price')
ax_price.legend(fontsize=9, loc='upper right')
ax_price.grid(alpha=0.3)

stats_box = ax_price.text(
    0.02, 0.97, summary_text(runs),
    transform=ax_price.transAxes, verticalalignment='top',
    fontsize=8.5, family='monospace',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
)

line_v, = ax_vol.plot(range(len(mean_v)), mean_v, color='coral', linewidth=1.5)
ax_vol.set_xlabel('Time Step')
ax_vol.set_ylabel('Avg Trades')
ax_vol.set_title('Trade Volume per Tick', fontsize=11)
ax_vol.grid(alpha=0.3)


def refresh_title(fee):
    ax_price.set_title(f'Price Trajectory  —  Fee: {fee*100:.0f}%', fontsize=12)


refresh_title(INITIAL_FEE)

# ── slider ──────────────────────────────────────────────────────────────────
ax_slider = plt.axes([0.15, 0.07, 0.70, 0.03])
slider = Slider(ax_slider, 'Transaction Fee (%)', 0, 100,
                valinit=INITIAL_FEE * 100, valstep=1, color='steelblue')


def on_change(val):
    fee = slider.val / 100.0
    runs = run_experiment(fee)
    mean_p, ci_p, mean_v = compute_trajectories(runs)
    steps = list(range(len(mean_p)))
    upper = [m + c for m, c in zip(mean_p, ci_p)]
    lower = [m - c for m, c in zip(mean_p, ci_p)]

    line_p.set_data(steps, mean_p)

    fill_ref[0].remove()
    fill_ref[0] = ax_price.fill_between(steps, lower, upper, color='steelblue', alpha=0.2)

    pad = max(upper) * 0.08
    ax_price.set_ylim(max(0, min(lower) - pad), max(upper) + pad)

    line_v.set_data(range(len(mean_v)), mean_v)
    ax_vol.set_ylim(0, max(mean_v) * 1.25 + 0.5)

    stats_box.set_text(summary_text(runs))
    refresh_title(fee)
    fig.canvas.draw_idle()


slider.on_changed(on_change)

plt.show()
