"""
Agent module: defines the trader types in the marketplace.

Each agent type implements decide(), returning 'buy', 'sell', or 'hold' based on the current market price and price history.
Agents are heterogeneous within types.
Parameters such as activity rate, target inventory, lookback window and reaction threshold are randomised at creation time.
"""
import random

class Agent:
    def __init__(self, agent_id, balance=1000.0, inventory=5):
        self.agent_id = agent_id
        self.balance = balance #Steam wallet balance
        self.inventory = inventory #number of items currently held

    def decide(self, market_price, price_history):
        raise NotImplementedError

#CasualTrader: barely engages with the market
#acts randomly when active, otherwise holds
#represents the "noise trader" archetype
class CasualTrader(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activity_rate = random.uniform(0.05, 0.15) #each casual has a slightly different activity rate (5%-15%)

    def decide(self, market_price, price_history):
        if random.random() < self.activity_rate:
            return random.choice(['buy', 'sell', 'hold'])
        return 'hold'

#Collector: patient buyer trying to complete a set
#buys when prices look reasonable relative to the recent average
#sometimes sells duplicates when over-stocked
class Collector(Agent):
    def __init__(self, *args, **kwargs):
        target_inventory = kwargs.pop('target_inventory', None)
        super().__init__(*args, **kwargs)
        #heterogeneity: each collector wants a different set size (8-15) and has a different willingness to overpay (2%-10% above recent mean)
        self.target_inventory = target_inventory or random.randint(8, 15)
        self.price_tolerance = random.uniform(1.02, 1.10)

    def decide(self, market_price, price_history):
        #buy if items are still needed, are affordable, and price is within personal tolerance band above the recent average
        if self.inventory < self.target_inventory and self.balance > market_price:
            recent_mean = self._mean_price(price_history)
            if market_price < recent_mean * self.price_tolerance:
                return 'buy'
            
        #small chance of selling duplicates if over-stocked
        if self.inventory > self.target_inventory and random.random() < 0.05:
            return 'sell'
        return 'hold'

    def _mean_price(self, price_history):
        if not price_history:
            return 1
        window = price_history[-20:]
        return sum(window) / len(window)

#Speculator: trend-follower
#buys when prices are rising, sells when falling
#reacts to percentage changes rather than absolute changes, so the strategy is consistent across different price levels
class Speculator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #heterogeneity: each speculator uses a different lookback window (3-10) and has a different threshold before reacting to a trend (0%-2%)
        self.lookback = random.randint(3, 10)
        self.threshold = random.uniform(0.0, 0.02)

    def decide(self, market_price, price_history):
        #need at least lookback+1 prices to compute a trend
        if len(price_history) < self.lookback + 1:
            return 'hold'

        #convert absolute trend to a relative (percentage) trend so behaviour is consistent regardless of price level
        recent_trend = price_history[-1] - price_history[-self.lookback]
        relative_trend = recent_trend / price_history[-self.lookback]

        if relative_trend > self.threshold and self.balance > market_price:
            return 'buy'
        elif relative_trend < -self.threshold and self.inventory > 0:
            return 'sell'
        return 'hold'