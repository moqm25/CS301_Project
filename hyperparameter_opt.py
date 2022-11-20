import group_10__semantic_segmentation_of_satellite_imagery as mdl
import numpy as np
import matplotlib as plt
import re
from tensorflow import keras

START_MODEL_COUNT = 4
TOTAL_ITER_COUNT = 24

dataset = mdl.getData()

def model_name(learning_rate, filter_count_factor):
    return f"model_{filter_count_factor}-{str(learning_rate)}"
def evaluate_model(learning_rate, filter_count_factor):
    history = mdl.train_model(learning_rate, filter_count_factor, model_name(learning_rate, filter_count_factor), dataset)
    keras.backend.clear_session()
    return history.history['loss'][-1]

### Search Space [28 x 6]###
learning_rate_space = []
for e in [0, -1, -2, -3, -4, -5, -6]:
    for d in [1, 0.75, 0.5, 0.25]:    
        learning_rate_space.append((1.0 * d) * (10 ** e))
print("Learning Rates: ", learning_rate_space)

filter_count_space = [16, 14, 12, 10, 8, 6]
print("Filter Counts: ", filter_count_space)

# [[learning_rate, filter_count_factor, loss], ...]
model_losses = []

#returns better_models, worse_models
def partition_existing_models(threshold=0.5):
    mls = model_losses[np.argsort(model_losses[:, 2])]
    threshold_point = int(len(mls) * threshold_point)
    return mls[:threshold_point], mls[threshold_point:]
#Provides a probability describing how likely it is for the given value to appear in the given set
def likelihood(newValue, existingSet):
    return np.exp(newValue) / sum(np.exp(existingSet))

def find_best_params():
    good_models, bad_models = partition_existing_models()
    lr_evals = likelihood(learning_rate_space, good_models[:, 0]) / likelihood(learning_rate_space, bad_models[:, 0])
    fcf_evals = likelihood(filter_count_space, good_models[:, 1]) / likelihood(filter_count_space, bad_models[:, 1])
    return learning_rate_space[np.argmax(lr_evals)], filter_count_space[np.argmax(fcf_evals)]

for _ in range(START_MODEL_COUNT):
    lr_arg = np.random.choice(range(len(learning_rate_space)))
    lr = learning_rate_space[lr_arg]

    fcf_arg = np.random.choice(range(len(filter_count_space)))
    fcf = filter_count_space[fcf_arg]

    print("RANDOM STAGE: Evaluating ", model_name(lr, fcf))
    model_losses.append([lr, fcf, evaluate_model(lr, fcf)])

for _ in range(TOTAL_ITER_COUNT - START_MODEL_COUNT):
    lr, fcf = find_best_params()

    print("TPE STAGE: Evaluating ", model_name(lr, fcf))
    model_losses.append([lr, fcf, evaluate_model(lr, fcf)])

best_model = model_losses[np.argsort(model_losses[:, 2])[0]]
print(f"\nBEST MODEL: {best_model}\nNAME: {model_name(best_model[0], best_model[1])}")