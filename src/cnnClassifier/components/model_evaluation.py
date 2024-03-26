import tensorflow as tf
from pathlib import Path
import mlflow
import shutil
import mlflow.keras
import numpy as np
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier import logger
from sklearn.metrics import confusion_matrix, classification_report

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _test_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

       

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255
        )

        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.testing_data,target_size=(224,224),class_mode='categorical',
            
            shuffle=False,
            **dataflow_kwargs
        )






    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        logger.info("==================Evaluation================")
        self.model = self.load_model(self.config.path_of_model)
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)
        self.save_score()

        predict=self.model.predict(self.test_generator)
        y_pred = np.argmax(predict, axis=1)
        cm = confusion_matrix(self.test_generator.classes, y_pred)
        logger.info(cm)
        logger.info(print(classification_report(self.test_generator.classes,y_pred)))
        
    def move_model(self):
        source_path=self.config.path_of_model
        destination_path=Path(r"C:\Users\judirispah\Chest-Disease-Classification-using-CT-scan\model")
        shutil.copy(source_path, destination_path)
    
    
    
        
    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme #http
        logger.info(tracking_url_type_store)
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
               
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")