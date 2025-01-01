import pickle

with open('C:/Users/Pinter Andrea/Documents/GitHub/ProjectQ/AgentBasedModeling/Replicator Dynamic Market '
          'Model/Model_basic.pkl', 'rb') as f:
    sim_data = pickle.load(f)

sim_data = sim_data[99]

comp_ms_value = [[comp['MarketShare'] for comp in step['Companies']] for step in sim_data]

print(comp_ms_value)
