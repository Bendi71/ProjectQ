import matplotlib.pyplot as plt
import numpy as np

from Market import Market

Market = Market()

Market.set_parameters(0.1, -0.5, 0.5, 0.005, 0.2, 0.1, 0.06, 10, 0.02)
Market.compile(1000, 100, 10)
Market.step()

reshaped_MS = Market.reshape_MS_data(Market.sim_all_companies[0], Market.steps)
reshaped_Q = Market.reshape_Q_data(Market.sim_all_companies[0], Market.steps)

# visualize market share
plt.figure(figsize=(32, 7))  # 1 row, 2 columns, 1st subplot
plt.subplot(1, 2, 1)
plt.plot(reshaped_MS.T)
plt.ylabel('Market Share')
plt.xlabel('Step')
plt.title('Market Share over Time')
plt.subplot(1, 2, 2)
plt.plot(reshaped_Q.T)
plt.ylabel('Quality')
plt.title('Quality over Time')
plt.xlabel('Step')

plt.show()

total_market_values = np.mean([[step['Total_Market'] for step in sim] for sim in Market.sim_data], axis=0)
av_quality_values = np.mean([[step['Av_Quality'] for step in sim] for sim in Market.sim_data], axis=0)
num_companies = np.mean([[step['Num_Companies'] for step in sim] for sim in Market.sim_data], axis=0)
plt.figure(figsize=(40, 7))
plt.subplot(1, 3, 1)
plt.plot(total_market_values, label='Total Market', color='blue')
plt.subplot(1, 3, 2)
plt.plot(av_quality_values, label='Average Quality', color='green')
plt.subplot(1, 3, 3)
plt.plot(num_companies)
plt.show()

# with open('C:/Users/Pinter Andrea/Documents/GitHub/ProjectQ/AgentBasedModeling/Replicator Dynamic Market '
# 'Model/Model_basic.pkl', 'wb') as f:
# pickle.dump(Market.sim_data, f)
