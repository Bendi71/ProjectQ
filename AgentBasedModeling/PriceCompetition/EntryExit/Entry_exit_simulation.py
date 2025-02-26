import pickle

import matplotlib.pyplot as plt
import numpy as np

from Market_entry_exit import Market

market = Market()

market.set_parameters(100, 10, 0.1, 0.4, 0.05, 10.0, exit_threshold=0.05)
market.compile(1000, 100)
market.run_simulation()

with open(
        '/AgentBasedModeling/PriceCompetition/EntryExit/Market_data.pkl',
        'wb') as f:
    pickle.dump(market.sim_data, f)

reshaped_market_share = market.reshape_MS_data(market.sim_all_sellers[0], market.steps)
reshaped_price = market.reshape_P_data(market.sim_all_sellers[0], market.steps)

# visualize market share for one simulation
plt.figure(figsize=(32, 7))  # 1 row, 2 columns, 1st subplot
plt.subplot(1, 2, 1)
plt.plot(reshaped_market_share.T)
plt.ylabel('Market Share')
plt.xlabel('Step')
plt.title('Market Share over Time')
plt.subplot(1, 2, 2)
plt.plot(reshaped_price.T)
plt.ylabel('Price')
plt.title('Price over Time')
plt.xlabel('Step')
plt.show()

# visualize market share for all simulations
avg_price = np.mean([[step['avg_price'] for step in sim] for sim in market.sim_data], axis=0)
avg_num_switch = np.mean([[step['num_switch'] for step in sim] for sim in market.sim_data], axis=0)
avg_num_sellers = np.mean([[step['num_sellers'] for step in sim] for sim in market.sim_data], axis=0)

plt.figure(figsize=(48, 7))  # 1 row, 2 columns, 1st subplot
plt.subplot(1, 3, 1)
plt.plot(avg_price)
plt.ylabel('Average Price')
plt.xlabel('Step')
plt.title('Average Price over Time')
plt.subplot(1, 3, 2)
plt.plot(avg_num_switch)
plt.ylabel('Number of Switching')
plt.xlabel('Step')
plt.title('Number of Switching over Time')
plt.subplot(1, 3, 3)
plt.plot(avg_num_sellers)
plt.ylabel('Number of Sellers')
plt.xlabel('Step')
plt.title('Number of Sellers over Time')
plt.show()
