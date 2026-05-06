"""
Simulation module: orchestrates the market and agent population across discrete time steps.

A simulation is built from a config dictionary.
- Includes agent counts, market parameters, run length
- Each tick, every agent decides on an action; the market matches buyers to sellers and updates the price
- The full price history is returned for downstream analysis
"""
from agents import CasualTrader, Collector, Speculator
from market import Market

class Simulation:
    def __init__(self, config):
        #total agent count is needed by the market for demand normalisation
        n_agents = (config.get('casual', 10)
                    + config.get('collectors', 5)
                    + config.get('speculators', 5))

        #build the market with calibration parameters from the config
        #defaults are used for any parameter not specified in the config
        self.market = Market(
            initial_price=config.get('initial_price', 10.0),
            n_agents=n_agents,
            demand_sensitivity=config.get('demand_sensitivity', 0.05), #previously 0.5, more realistic for a high-frequenct simulation
            noise_std=config.get('noise_std', 0.01),
            transaction_fee=config.get('transaction_fee', 0.0),
        )

        self.agents = self._init_agents(config)
        self.steps = config.get('steps', 200)

    #helper: build the population of agents based on counts in the config
    def _init_agents(self, config):
        agents = []
        aid = 0
        for _ in range(config.get('casual', 10)):
            agents.append(CasualTrader(aid)); aid += 1
        for _ in range(config.get('collectors', 5)):
            agents.append(Collector(aid)); aid += 1
        for _ in range(config.get('speculators', 5)):
            agents.append(Speculator(aid)); aid += 1
        return agents

    #main simulation loop: one iteration represents one round of market activity
    def run(self):
        for step in range(self.steps):
            orders = []
            for agent in self.agents:
                action = agent.decide(
                    self.market.price,
                    self.market.price_history,
                    self.market.transaction_fee #pass fee so agents can respond to it
                )
                if action in ('buy', 'sell'):
                    orders.append((action, agent))
            self.market.execute(orders)
        return self.market.price_history