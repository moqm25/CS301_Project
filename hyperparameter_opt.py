import simple_multi_unet_model as unet
import group_10__semantic_segmentation_of_satellite_imagery as mdl
#import nni
from tensorflow.keras.metrics import MeanIoU
from tensorflow import keras
import tensorflow as tf

params = {
    'filter_count_factor': 16,
    'learning_rate': 0.001,
}

#optimized_params = nni.get_next_parameter()
#params.update(optimized_params)

model= unet.multi_unet_model(params['filter_count_factor'], params['learning_rate'])

x_train, x_test, y_train, y_test = mdl.getData()

callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: nni.report_intermediate_result(logs['accuracy'])
)

model.fit(x_train, y_train, epochs=5, verbose=2, callbacks=[callback])
evaluation = model.evaluate(x_test, y_test, verbose=2, return_dict=True)

#nni.report_final_result(evaluation['loss'])