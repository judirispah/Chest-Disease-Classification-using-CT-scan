import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from keras.applications.vgg16 import preprocess_input
#from cnnClassifier.entity.config_entity import TrainingConfig



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
        
 

    
    def predict(self):
        ## load model
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(r"C:\Users\judirispah\Chest-Disease-Classification-using-CT-scan\artifacts\training\model.h5")

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = preprocess_input(test_image)
        test_image=test_image/225
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result == 0:
            prediction = 'Adenocarcinoma Cancer'
            return [{ "image" : prediction}]

        if result == 1:
            prediction = 'Covid'
            return [{ "image" : prediction}]
        
        
        if result == 2:
            prediction = 'Large cell carcinoma cancer'
            return [{ "image" : prediction}]
        
        if result == 3:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        
        if result == 4:
            prediction = 'Pneumonia'
            return [{ "image" : prediction}]
        
        if result == 5:
            prediction = 'Squamous carcinoma Cancer'
            return [{ "image" : prediction}]