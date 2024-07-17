import numpy as np
from Market import Market
import pickle
import matplotlib.pyplot as plt

Market = Market()

Market.set_parameters(0.1, -0.5, 0.5, 0.005, 0.2, 0.1, 0.06, 10, 0.01)
Market.compile(100, 1, 10)
Market.step()

print([[comp.History, [comp.entry_step,comp.exit_step]] for comp in Market.all_companies])

#total_market_values = np.mean([[step['Total_Market'] for step in sim] for sim in Market.sim_data[0]], axis=0)
#plt.plot(total_market_values, label='Total Market', color='blue')
#plt.show()

#with open('C:/Users/Pinter Andrea/Documents/GitHub/ProjectQ/Agent based modeling/Replicator Dynamic Market '
          #'Model/Model_basic.pkl', 'wb') as f:
    #pickle.dump(Market.sim_data, f)