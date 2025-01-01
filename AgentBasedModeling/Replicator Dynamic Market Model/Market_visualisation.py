import pickle

import matplotlib.pyplot as plt
import numpy as np

# importing the data
with open('C:/Users/Pinter Andrea/Documents/GitHub/ProjectQ/AgentBasedModeling/Replicator Dynamic Market '
          'Model/Model_basic.pkl', 'rb') as f:
    sim_data = pickle.load(f)

total_market_values = np.mean([[step['Total_Market'] for step in sim] for sim in sim_data], axis=0)
av_quality_values = np.mean([[step['Av_Quality'] for step in sim] for sim in sim_data], axis=0)
num_companies = np.mean([[step['Num_Companies'] for step in sim] for sim in sim_data], axis=0)
steps = range(len(sim_data[0]))

# Plotting Total Market over time
plt.figure(figsize=(24, 7))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(steps, total_market_values, label='Total Market', color='blue')
# add a second plot to the same subplot with different axis
plt.twinx()
plt.plot(steps, num_companies, label='Number of Companies', color='red')
plt.ylabel('Number of Companies')
plt.title('Total Market and Number of Companies over Time')
plt.xlabel('Step')
plt.ylabel('Total Market')
plt.legend()

# Plotting Average Quality over time
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(steps, av_quality_values, label='Average Quality', color='green')
plt.xlabel('Step')
plt.ylabel('Average Quality')
plt.title('Average Quality over Time')
plt.legend()

plt.tight_layout()
plt.show()
