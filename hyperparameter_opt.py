import group_10__semantic_segmentation_of_satellite_imagery as mdl
import numpy as np
import matplotlib as plt

CLASS_COUNT = 6

trainset_sequence = ["t"]

def evaluate_model(learning_rate, filter_count_factor):
    history = mdl.train_model(learning_rate, filter_count_factor, f"model_{filter_count_factor}-{str(learning_rate)}")
    return history['loss'][-1]

### Search Space ###
learning_rate_space = []
for e in [0, -1, -2, -3, -4, -5, -6]:
    for d in [1, 0.75, 0.5, 0.25]:    
        learning_rate_space.append((1.0 / d) * (10 ** e))
print("Learning Rates: ", learning_rate_space)

filter_count_space = list(set(np.linspace(2, 16, dtype=int)))
print("Filter Counts: ", filter_count_space)

