from tensorflow import keras
import simple_multi_unet_model as unet_model
import group_10__semantic_segmentation_of_satellite_imagery as mdl
import random
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("distilled_model.hdf5",
                    custom_objects={'dice_loss_plus_1focal_loss': unet_model.get_total_loss(),
                                    'jacard_coef':unet_model.jacard_coef})

old_model = keras.models.load_model("models/final_model.hdf5",
                    custom_objects={'dice_loss_plus_1focal_loss': unet_model.get_total_loss(),
                                    'jacard_coef':unet_model.jacard_coef})

x_train, x_test, y_train, y_test = mdl.getData()
# # #IOU
# y_pred=model.predict(X_test)
# y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_train, axis=3)

continueImgView = input("View image?")
while (continueImgView == 'y'):
    
    test_img_number = random.randint(0, len(x_train))
    test_img = x_train[test_img_number]
    ground_truth=y_test_argmax[test_img_number]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    old_prediction = (old_model.predict(test_img_input))
    old_predicted_img=np.argmax(old_prediction, axis=3)[0,:,:]

    

    plt.figure(figsize=(16, 8))
    plt.subplot(241)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.subplot(242)
    plt.title('Testing Label')
    plt.imshow(ground_truth)
    plt.subplot(243)
    plt.title('Old Prediction on test image')
    plt.imshow(old_predicted_img)
    plt.subplot(244)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img)
    plt.show()

    #####################################################################
    continueImgView = input("Continue viewing?")