#this file runs the simulation

import csv
from agents import CasualTrader, Collector, Speculator
from market import Market

#this class builds the market and the population, runs the clock and saves the results
class Simulation:
    def __init__(self, config):
        self.market = Market(initial_price=config.get('initial_price', 10.0))
        self.agents = self._init_agents(config)
        self.steps = config.get('steps', 200)

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

    def run(self):
        for step in range(self.steps):
            orders = []
            for agent in self.agents:
                action = agent.decide(self.market.price, self.market.price_history)
                if action in ('buy', 'sell'):
                    orders.append((action, agent))
            self.market.execute(orders)
        return self.market.price_history