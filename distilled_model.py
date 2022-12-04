# https://youtu.be/jvZm8REF2KY
"""
Standard Unet
Model not compiled here, instead will be done externally to make it
easy to test various loss functions and optimizers. 
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
import segmentation_models as sm
import tensorflow as tf
import numpy as np

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def total_loss():
    weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
    dice_loss = sm.losses.DiceLoss(class_weights=weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    return total_loss

################################################################
def multi_unet_model(filter_count_factor = 16, learning_rate = 0.001, n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(filter_count_factor, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(filter_count_factor, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(2 * filter_count_factor, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(2 * filter_count_factor, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    
    #Expansive path 
     
    u9 = Conv2DTranspose(filter_count_factor, (2, 2), strides=(2, 2), padding='same')(c2)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(filter_count_factor, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(filter_count_factor, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
    dice_loss = sm.losses.DiceLoss(class_weights=weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    
    prob_thresholds = np.linspace(0, 1, num=1000).tolist()
    metrics = ['accuracy', tf.keras.metrics.Precision(name="precision", thresholds=prob_thresholds), tf.keras.metrics.Recall(name="recall", thresholds=prob_thresholds)]


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=total_loss, metrics=metrics)
    return model