"""
Test suite for the agent-based virtual goods marketplace simulation.

Run with:    pytest test_simulation.py -v
Run with:    pytest test_simulation.py -v --tb=short

The suite contains 18 tests grouped into four sections:
  - Agent behaviour (7 tests)
  - Market mechanics (6 tests)
  - Simulation orchestration (3 tests)
  - Validation utilities (2 tests)

Tests cover correctness of agent decision rules, market price formation
and trade execution, and integration-level behaviour of the simulation
as a whole. Stochastic tests use fixed seeds for reproducibility.
"""
import random
import numpy as np
import pytest

from agents import CasualTrader, Collector, Speculator
from market import Market
from simulation import Simulation
from stylised_facts import log_returns, autocorrelation

#section 1: agent behaviour
#tests that each agent type returns sensible actions under controlled conditions
def test_casual_trader_returns_valid_action():
    random.seed(42)
    agent = CasualTrader(agent_id=0)

    #run many decisions to cover both active and inactive casual trader behaviour
    actions = [agent.decide(market_price=10.0, price_history=[10.0]) for _ in range(200)]

    assert all(a in ('buy', 'sell', 'hold') for a in actions)

def test_collector_buys_below_fair_value():
    random.seed(0)
    collector = Collector(agent_id=1)

    #force inventory below target so the buy condition can trigger
    collector.inventory = 0
    collector.balance = 1000.0

    #price well below fair value should trigger a buy
    action = collector.decide(market_price=5.0, price_history=[10.0])

    assert action == 'buy'

def test_collector_does_not_sell_with_zero_inventory():
    random.seed(0)
    collector = Collector(agent_id=2)
    collector.inventory = 0

    #even at a very high price, an agent with no inventory cannot sell
    action = collector.decide(market_price=1000.0, price_history=[10.0])

    assert action != 'sell'

def test_collector_holds_when_fee_suppresses_effective_return():
    random.seed(0)
    collector = Collector(agent_id=3)

    #force a deterministic sell threshold so the fee effect is easy to test
    collector.inventory = 5
    collector.fair_value = 10.0
    collector.sell_tolerance = 1.2

    #without fees, price 13 gives a return above the sell threshold of 12
    #with a 15% fee, effective return is 13 * 0.85 = 11.05, so selling is not worth it
    action = collector.decide(
        market_price=13.0,
        price_history=[10.0],
        transaction_fee=0.15
    )

    assert action == 'hold'

def test_speculator_holds_when_history_too_short():
    random.seed(0)
    speculator = Speculator(agent_id=4)

    #history is shorter than the minimum possible lookback window, so no trend can be computed
    short_history = [10.0, 10.1]
    action = speculator.decide(market_price=10.1, price_history=short_history)

    assert action == 'hold'

def test_speculator_buys_on_strong_uptrend():
    random.seed(0)
    speculator = Speculator(agent_id=5)

    #force deterministic trend parameters so the test does not depend on random heterogeneity
    speculator.lookback = 3
    speculator.threshold = 0.01
    speculator.balance = 1000.0

    #strong uptrend: lookback compares 11 to 15, so (15 - 11) / 11 = 0.364
    rising_history = [10.0, 11.0, 12.0, 15.0]
    action = speculator.decide(market_price=15.0, price_history=rising_history)

    assert action == 'buy'

def test_speculator_sells_on_strong_downtrend():
    random.seed(0)
    speculator = Speculator(agent_id=6)

    #force deterministic trend parameters so the sell branch is directly tested
    speculator.lookback = 3
    speculator.threshold = 0.01
    speculator.inventory = 5

    #strong downtrend: lookback compares 14 to 10, so (10 - 14) / 14 = -0.286
    falling_history = [15.0, 14.0, 12.0, 10.0]
    action = speculator.decide(market_price=10.0, price_history=falling_history)

    assert action == 'sell'

#section 2: market mechanics
#tests price updates, trade matching, fees, volume records, and affordability checks
def test_market_starts_at_initial_price():
    market = Market(initial_price=10.0)

    assert market.price == 10.0
    assert market.price_history == [10.0]

def test_market_price_never_negative():
    random.seed(0)
    market = Market(initial_price=10.0, n_agents=20, demand_sensitivity=10.0)

    #construct an extreme imbalance: many sellers, no buyers
    sellers = [Collector(i) for i in range(15)]
    for s in sellers:
        s.inventory = 5

    orders = [('sell', s) for s in sellers]

    #repeat the imbalance for many ticks to test the price floor
    for _ in range(50):
        market.execute(orders)

    assert market.price >= 0.01
    assert all(p >= 0.01 for p in market.price_history)

def test_market_matches_minimum_of_buyers_and_sellers():
    random.seed(0)
    market = Market(initial_price=10.0)

    buyers = [Collector(i) for i in range(5)]
    sellers = [Collector(i + 100) for i in range(3)]

    #ensure all buyers can afford the trade and all sellers have inventory
    for b in buyers:
        b.balance = 100.0
        b.inventory = 0

    for s in sellers:
        s.inventory = 5

    orders = [('buy', b) for b in buyers] + [('sell', s) for s in sellers]
    market.execute(orders)

    #only the smaller side of the book can be matched
    assert market.trades_executed[-1] == 3

def test_broke_buyer_is_filtered_out():
    random.seed(0)
    market = Market(initial_price=10.0)

    buyer = Collector(1)
    buyer.balance = 0.0
    buyer.inventory = 0

    seller = Collector(2)
    seller.balance = 0.0
    seller.inventory = 5

    orders = [('buy', buyer), ('sell', seller)]
    market.execute(orders)

    #buyer cannot afford the item, so no trade should execute even though a seller exists
    assert market.trades_executed[-1] == 0
    assert buyer.inventory == 0
    assert seller.inventory == 5
    assert buyer.balance == 0.0
    assert seller.balance == 0.0

def test_transaction_fee_deducted_from_seller():
    random.seed(0)
    market = Market(initial_price=100.0, transaction_fee=0.15)

    buyer = Collector(1)
    buyer.balance = 1000.0
    buyer.inventory = 0

    seller = Collector(2)
    seller.balance = 0.0
    seller.inventory = 5

    orders = [('buy', buyer), ('sell', seller)]
    market.execute(orders)

    #seller receives price after the 15% platform fee: 100 * 0.85 = 85
    assert seller.balance == pytest.approx(85.0, rel=1e-6)

    #buyer pays the full listed market price
    assert buyer.balance == pytest.approx(900.0, rel=1e-6)

    #platform keeps the fee difference
    assert market.fees_collected == pytest.approx(15.0, rel=1e-6)

def test_volume_recorded_each_tick():
    market = Market(initial_price=10.0)
    initial_length = len(market.trades_executed)

    #run three empty ticks to test that volume is still recorded when no trades occur
    for _ in range(3):
        market.execute(orders=[])

    assert len(market.trades_executed) == initial_length + 3
    assert all(v == 0 for v in market.trades_executed[-3:])

#section 3: simulation orchestration
#tests that the simulation creates agents correctly and produces reproducible histories
def test_simulation_produces_correct_history_length():
    random.seed(0)
    config = {'casual': 5, 'collectors': 5, 'speculators': 2, 'steps': 50}

    sim = Simulation(config)
    history = sim.run()

    #N simulation steps produce N+1 prices because the initial price is included
    assert len(history) == 51

def test_simulation_creates_correct_agent_counts():
    config = {'casual': 8, 'collectors': 8, 'speculators': 4, 'steps': 10}

    sim = Simulation(config)

    n_casual = sum(1 for a in sim.agents if isinstance(a, CasualTrader))
    n_collectors = sum(1 for a in sim.agents if isinstance(a, Collector))
    n_speculators = sum(1 for a in sim.agents if isinstance(a, Speculator))

    assert n_casual == 8
    assert n_collectors == 8
    assert n_speculators == 4
    assert len(sim.agents) == 20

def test_simulation_is_reproducible_with_same_seed():
    config = {'casual': 5, 'collectors': 5, 'speculators': 2, 'steps': 100}

    #same seed should reproduce the exact same stochastic market path
    random.seed(123)
    history_a = Simulation(config).run()

    random.seed(123)
    history_b = Simulation(config).run()

    assert history_a == history_b

    #different seed should usually produce a different market path
    random.seed(456)
    history_c = Simulation(config).run()

    assert history_a != history_c

#section 4: validation utilities
#tests small statistical helper functions used by stylised_facts.py
def test_log_returns_correct_length_and_value():
    prices = [10.0, 20.0, 10.0]
    returns = log_returns(prices)

    assert len(returns) == 2

    #ln(20 / 10) = ln(2), then ln(10 / 20) = -ln(2)
    assert returns[0] == pytest.approx(np.log(2), rel=1e-6)
    assert returns[1] == pytest.approx(-np.log(2), rel=1e-6)

def test_autocorrelation_of_constant_series_is_zero_or_safe():
    constant = [5.0] * 20
    result = autocorrelation(constant, lag=1)

    #constant series has zero variance, so the helper safely returns 0 instead of crashing
    assert result == 0.0

    #a smooth increasing series should have very strong positive lag-1 autocorrelation
    linear = list(range(50))
    rho = autocorrelation(linear, lag=1)

    assert rho > 0.9