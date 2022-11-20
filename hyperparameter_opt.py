import group_10__semantic_segmentation_of_satellite_imagery as mdl
import numpy as np
import matplotlib as plt
import re
from tensorflow import keras
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def resetSession():
    keras.backend.clear_session()
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
START_MODEL_COUNT = 4
TOTAL_ITER_COUNT = 24

dataset = mdl.getData()

def model_name(learning_rate, filter_count_factor):
    return f"model__{filter_count_factor}_{str(learning_rate)}"
def evaluate_model(learning_rate, filter_count_factor):
    history = mdl.train_model(learning_rate, filter_count_factor, model_name(learning_rate, filter_count_factor), dataset)
    resetSession()
    return history.history['loss'][-1]

### Search Space [28 x 6]###
learning_rate_space = []
for e in [-2, -3, -4]:
    for d in [1, 0.75, 0.5, 0.25]:    
        learning_rate_space.append((1.0 * d) * (10 ** e))
print("Learning Rates: ", learning_rate_space)

filter_count_space = [16, 14, 12, 10, 8, 6]
print("Filter Counts: ", filter_count_space)

# [[learning_rate, filter_count_factor, loss], ...]
model_losses = []

#returns better_models, worse_models
def partition_existing_models(threshold=0.5):
    mls = np.array(model_losses)[np.argsort(np.array(model_losses)[:, 2])]
    threshold_point = int(len(mls) * threshold)
    return mls[:threshold_point], mls[threshold_point:]
#Provides a probability describing how likely it is for the given value to appear in the given set
def likelihood(newValue, existingSet):
    return np.exp(newValue) / sum(np.exp(existingSet))

def find_best_params():
    good_models, bad_models = partition_existing_models()
    lr_evals = np.argsort(likelihood(learning_rate_space, good_models[:, 0]) / likelihood(learning_rate_space, bad_models[:, 0]))
    fcf_evals = np.argsort(likelihood(filter_count_space, good_models[:, 1]) / likelihood(filter_count_space, bad_models[:, 1]))
    def getValuesAt(i):
        return [learning_rate_space[lr_evals[-max(i, len(learning_rate_space))]], filter_count_space[fcf_evals[-max(i, len(filter_count_space))]]]
    outputValue = getValuesAt(1)
    i = 1
    while outputValue in np.array(model_losses)[:, 0:1]:
        i += 1
        outputValue = getValuesAt(i)
    return outputValue


if os.path.isdir('models'):
    for file in os.listdir('models'):
        if str(file).startswith('model_'):
            print(f"Loading cached model {file}...")
            params = np.array(file.split('__')[1].split('.hdf5')[0].split('_')).astype(np.float)
            model_losses.append([params[1], params[0], params[2]])


for _ in range(max(START_MODEL_COUNT - len(model_losses), 0)):
    lr_arg = np.random.choice(range(len(learning_rate_space)))
    lr = learning_rate_space[lr_arg]

    fcf_arg = np.random.choice(range(len(filter_count_space)))
    fcf = filter_count_space[fcf_arg]

    print("RANDOM STAGE: Evaluating ", model_name(lr, fcf))
    model_losses.append([lr, fcf, evaluate_model(lr, fcf)])

for _ in range(max(TOTAL_ITER_COUNT - START_MODEL_COUNT - len(model_losses), 4)):
    lr, fcf = find_best_params()

    print("TPE STAGE: Evaluating ", model_name(lr, fcf))
    model_losses.append([lr, fcf, evaluate_model(lr, fcf)])

best_model = model_losses[np.argsort(model_losses[:, 2])[0]]
print(f"\nBEST MODEL: {best_model}\nNAME: {model_name(best_model[0], best_model[1])}")