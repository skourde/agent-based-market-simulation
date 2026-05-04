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
- result: collector-dominated stabilises, speculator-dominated bubbles, baseline shows mild drift
- three regimes produce distinct dynamics