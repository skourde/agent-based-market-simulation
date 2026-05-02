#this file handles the price and trades

import random

#this class keeps track of the current price and the history of past prices
#the price is updated based on the net demand from buy/sell orders and some random noise to simulate market unpredictability
#trades are executed by adjusting the agent's balance and inventory based on their actions and the current price
class Market:
    def __init__(self, initial_price=10.0):
        self.price = initial_price
        self.price_history = [initial_price]

    def execute(self, orders):
        buys = sum(1 for action, _ in orders if action == 'buy')
        sells = sum(1 for action, _ in orders if action == 'sell')
        
        net_demand = buys - sells
        price_change = net_demand * 0.02 * self.price
        
        noise = random.gauss(0, 0.01 * self.price)
        self.price = max(0.01, self.price + price_change + noise)
        
        for action, agent in orders:
            if action == 'buy' and agent.balance >= self.price:
                agent.balance -= self.price
                agent.inventory += 1
            elif action == 'sell' and agent.inventory > 0:
                agent.balance += self.price
                agent.inventory -= 1

        self.price_history.append(self.price)