#Calibration journey (for Methodology/Discussion)

#v1: naive linear price model
- price_change = net_demand × 0.02 × price (multiplicative feedback)
- no buyer-seller matching
- result: damped oscillations, items appearing from nowhere

#v2: log-returns + matching, but demand_sensitivity too high
- demand_sensitivity = 0.5
- result: runaway exponential growth, prices reaching ~£2800

#v3: calibrated demand sensitivity, no fair-value anchor
- demand_sensitivity = 0.05, speculator threshold raised
- result: still smooth exponential growth — collectors had no anchor

#v4: fair-value anchor for collectors
- collectors now buy near fair value, sell well above it
- result: collector-dominated stabilises, speculator-dominated bubbles, baseline still drifts due to insufficient anchoring force

#v5: rebalanced baseline composition (8 casual / 8 collectors / 4 speculators)
- increased collector presence in baseline to strengthen anchoring force
- reduced steps from 300 to 200 to stay within plausible price bounds
- result: baseline stabilises near £14, collector-dominated near £12, speculator-dominated shows clear bubble-and-crash to ~£800
- three regimes produce clearly distinct, explainable dynamics
- final parameters: demand_sensitivity=0.05, noise_std=0.01, steps=200