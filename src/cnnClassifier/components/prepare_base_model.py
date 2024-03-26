import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
from cnnClassifier import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=(224,224,3),
            weights=self.config.params_weights,
            include_top=False
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
        


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        batch=tf.keras.layers.BatchNormalization()(flatten_in)
        


        
        dense=tf.keras.layers.Dense(units=359, activation='relu',kernel_initializer=tf.keras.initializers.HeUniform(),use_bias=True,bias_initializer='zeros')(batch)
        batch=tf.keras.layers.BatchNormalization()(dense)
        dropout=tf.keras.layers.Dropout(0.5)(batch)

        dense=tf.keras.layers.Dense(150,activation="relu",)(dropout)
        batch=tf.keras.layers.BatchNormalization()(dense)
        dropout=tf.keras.layers.Dropout(0.5)(batch)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(dropout)
       

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        logger.info(model.input)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        logger.info(full_model.summary())
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)