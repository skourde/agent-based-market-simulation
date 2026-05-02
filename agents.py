#this file defines the different types of traders in the marketplace

import random

#base class for all agents where each agent has an ID, balance and inventory
class Agent:
    def __init__(self, agent_id, balance=1000.0, inventory=5):
        self.agent_id = agent_id
        self.balance = balance
        self.inventory = inventory

    #decide() method - should they buy, sell or hold?
    def decide(self, market_price, price_history):
        raise NotImplementedError

#random low-effort trader who buys/sells with 10% probability
#represents users who hoard items for fun or nostalgia, not profit
class CasualTrader(Agent):
    def decide(self, market_price, price_history):
        if random.random() < 0.1:
            return random.choice(['buy', 'sell', 'hold'])
        return 'hold'

#represents the patient users whose aim is to collect items
#they buy when prices are low and hold until they have a complete set, then stop trading
class Collector(Agent):
    def __init__(self, *args, target_inventory=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_inventory = target_inventory

    def decide(self, market_price, price_history):
        if self.inventory < self.target_inventory and self.balance > market_price:
            if market_price < self._mean_price(price_history) * 1.05:
                return 'buy'
        return 'hold'

    def _mean_price(self, price_history):
        return sum(price_history[-20:]) / len(price_history[-20:]) if price_history else 1

#represents the profit-seeking users who try to time the market
class Speculator(Agent):
    def decide(self, market_price, price_history):
        if len(price_history) < 5:
            return 'hold'
        recent_trend = price_history[-1] - price_history[-5]

        if recent_trend > 0 and self.balance > market_price:
            return 'buy'
        elif recent_trend < 0 and self.inventory > 0:
            return 'sell'
        return 'hold'