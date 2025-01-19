import matplotlib.pyplot as plt
import numpy as np

from Market import Market

market = Market()
market.set_parameters(7, 10, 0.1, 0.1)
market.compile(100, 10)
market.run_simulation()

consumer_surplus = np.mean([[step['consumer_surplus'] for step in sim] for sim in market.sim_data], axis=0)
seller_surplus = np.mean([[step['seller_surplus'] for step in sim] for sim in market.sim_data], axis=0)

av_price = np.mean([[step['transaction_price'] for step in sim] for sim in market.sim_data], axis=0)
av_consumer_price = np.mean([[step['transaction_price'] for step in sim] for sim in market.sim_data], axis=0)
av_max_price = np.mean([[step['max_price'] for step in sim] for sim in market.sim_data], axis=0)
av_min_price = np.mean([[step['min_price'] for step in sim] for sim in market.sim_data], axis=0)

plt.figure(figsize=(48, 7))  # 1 row, 2 columns, 1st subplot
plt.subplot(1, 2, 1)
plt.plot(consumer_surplus)
plt.plot(seller_surplus)
plt.legend(["Consumer surplus", "Seller surplus"])
plt.ylabel('Surplus')
plt.xlabel('Step')
plt.title('Average Consumer and Seller Surplus over Time')
plt.subplot(1, 2, 2)
plt.plot(av_price)
plt.plot(av_max_price)
plt.plot(av_min_price)
plt.ylabel('Price')
plt.xlabel('Step')
plt.title('Average Transaction Price over Time')
plt.show()
