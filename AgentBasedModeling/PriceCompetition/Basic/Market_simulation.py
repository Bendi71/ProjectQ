import pickle

import matplotlib.pyplot as plt
import numpy as np

from Market import Market

market = Market()

market.set_parameters(100, 3, 0.1, 0.4, 0.02, 10.0)
market.compile(100, 50)
market.run_simulation()

with open('/AgentBasedModeling/PriceCompetition/Basic/Market_data.pkl',
          'wb') as f:
    pickle.dump(market.sim_data, f)

# visualize market share for one simulation
market_shares = np.array([step['Market Shares'] for step in market.sim_data[0]])
prices = np.array([step['Prices'] for step in market.sim_data[0]])

plt.figure(figsize=(32, 7))  # 1 row, 2 columns, 1st subplot
plt.subplot(1, 2, 1)
plt.plot(market_shares)
plt.ylabel('Market Share')
plt.xlabel('Step')
plt.title('Market Share over Time')
plt.subplot(1, 2, 2)
plt.plot(prices)
plt.ylabel('Price')
plt.title('Price over Time')
plt.xlabel('Step')
plt.show()

# visualize market share for all simulations
avg_price = np.mean([[step['Average Price'] for step in sim] for sim in market.sim_data], axis=0)
avg_num_switch = np.mean([[step['Number Switching'] for step in sim] for sim in market.sim_data], axis=0)

plt.figure(figsize=(32, 7))  # 1 row, 2 columns, 1st subplot
plt.subplot(1, 2, 1)
plt.plot(avg_price)
plt.ylabel('Average Price')
plt.xlabel('Step')
plt.title('Average Price over Time')
plt.subplot(1, 2, 2)
plt.plot(avg_num_switch)
plt.ylabel('Number of Switching')
plt.xlabel('Step')
plt.title('Number of Switching over Time')
plt.show()
