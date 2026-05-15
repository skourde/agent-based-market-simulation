"""
Market module: marketplace simulation for the agent-based virtual goods

The Market class handles price formation and trade execution per simulation tick.
- Agents submit buy/sell orders, and the market matches viable buyers to sellers
- Trades are executed with the platform's transaction fee deducted from the seller's proceeds
- The price is updated using a log-return based on the imbalance between buy and sell intent
- Log-returns are used to avoid multiplicative feedback
"""
import random
import math

class Market:
    def __init__(self, initial_price=10.0, n_agents=20, demand_sensitivity=0.5,
                 noise_std=0.01, transaction_fee=0.0):
        #core market state - changes as the simulation runs
        self.price = initial_price #current market price
        self.initial_price = initial_price #starting price        
        self.price_history = [initial_price] #list of every price the item has had

        #calibration parameters - control how the market behaves
        self.n_agents = n_agents #total players in the simulation
        self.demand_sensitivity = demand_sensitivity #how strongly demand moves price
        self.noise_std = noise_std #background market noise level
        self.transaction_fee = transaction_fee #platform's cut (e.g. 0.15 = Steam's 15%)

        #tracking variables for analysis
        self.trades_executed = [] #number of trades per tick (volume series)
        self.trade_log = [] #individual trades (buyer_id, seller_id, price)
        self.fees_collected = 0.0 #running total of platform revenue

    #runs once per simulation tick to match orders, execute trades, and update price
    def execute(self, orders):
        #filter out buyers who can't afford and sellers who have no stock
        buyers = [a for action, a in orders if action == 'buy' and a.balance >= self.price]
        sellers = [a for action, a in orders if action == 'sell' and a.inventory > 0]

        #shuffle so every buyer has an equal chance of being matched, and vice versa
        random.shuffle(buyers)
        random.shuffle(sellers)

        #matching rule: only the smaller group's worth of trades can execute
        num_trades = min(len(buyers), len(sellers))
        for i in range(num_trades):
            buyer = buyers[i]
            seller = sellers[i]

            #buyer pays the full listed price; seller receives price minus the platform fee
            fee = self.price * self.transaction_fee
            buyer.balance -= self.price
            buyer.inventory += 1
            seller.balance += self.price - fee
            seller.inventory -= 1

            #update tracking
            self.fees_collected += fee
            self.trade_log.append((buyer.agent_id, seller.agent_id, self.price))

        #record trade volume for this tick
        self.trades_executed.append(num_trades)

        #update price based on the imbalance of intent so unfilled demand still pushes price
        #normalised by population so results don't depend mechanically on agent count
        net_demand = (len(buyers) - len(sellers)) / self.n_agents
        noise = random.gauss(0, self.noise_std)
        log_return = self.demand_sensitivity * net_demand + noise

        #apply the percentage change
        #max() floor stops price ever reaching zero.
        self.price = max(0.01, self.price * math.exp(log_return))
        self.price_history.append(self.price)