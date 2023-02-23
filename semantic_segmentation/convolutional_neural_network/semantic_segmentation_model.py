import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from tensorflow import keras
from tensorflow.keras import backend as k

from semantic_segmentation.convolutional_neural_network.backbones.deeplabv3 import Deeplabv3

import pickle
# import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


from semantic_segmentation.convolutional_neural_network.backbone_handler import BackboneHandler
from semantic_segmentation.convolutional_neural_network.data_generator import DataGenerator

from semantic_segmentation.preprocessing.augmentor import Augmentor
from semantic_segmentation.preprocessing.preprocessor import Preprocessor

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     physical_devices[0],
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
    # )

except Exception as e:
    print(e)
    print("ATTENTION: GPU IS NOT USED....")


class SemanticSegmentationModel:
    def __init__(self, model_folder, cfg):
        self.model_folder = model_folder

        self.color_coding = cfg.color_coding
        self.input_shape = cfg.opt["input_shape"]
        self.backbone = cfg.opt["backbone"]
        if "loss" in cfg.opt:
            self.loss_type = cfg.opt["loss"]
        else:
            self.loss_type = "binary_crossentropy"

        if "label_prep" in cfg.opt:
            self.label_prep = cfg.opt["label_prep"]
        else:
            self.label_prep = "basic"

        if "logistic" in cfg.opt:
            self.logistic = cfg.opt["logistic"]
        else:
            self.logistic = "sigmoid"
        self.model = None

        optimizer = None
        if "optimizer" in cfg.opt:
            if "adam" == cfg.opt["optimizer"]:
                optimizer = optimizers.Adam(lr=cfg.opt["init_learning_rate"])
            # elif "lazy_adam" == cfg.opt["optimizer"]:
            #     optimizer = tfa.optimizers.LazyAdam(lr=cfg.opt["init_learning_rate"])
            # elif "ranger" == cfg.opt["optimizer"]:
            #     optimizer = tfa.optimizers.RectifiedAdam(learning_rate=cfg.opt["init_learning_rate"])
        if optimizer is None:
            optimizer = optimizers.Adam(lr=cfg.opt["init_learning_rate"])
        self.optimizer = optimizer

        if "batch_size" in cfg.opt:
            self.batch_size = cfg.opt["batch_size"]
        else:
            self.batch_size = 1
        self.epochs = 1000

    def predict_tag(self, tag, confidence_threshold):
        return self.predict(tag.load_data(), confidence_threshold)

    def predict(self, data, confidence_threshold):
        y_pred = self.inference(data)
        y_pred = y_pred[0, :, :, :]
        h, w = y_pred.shape[:2]
        color_map = np.zeros((h, w, 3))
        for i, cls in enumerate(self.color_coding):
            idxs = np.where(y_pred[:, :, i] > confidence_threshold)
            color_map[idxs[0], idxs[1], :] = self.color_coding[cls][1]
        return color_map

    def inference(self, data):
        preprocessor = Preprocessor(self.input_shape)
        img = preprocessor.apply(data)
        img = np.expand_dims(img, axis=0)
        res = self.model.predict_on_batch(img)
        return res

    def build(self, compile_model=True):
        num_classes = len(self.color_coding)
        input_layer = Input(batch_shape=(None,
                                         self.input_shape[0],
                                         self.input_shape[1],
                                         self.input_shape[2],
                                         ))
        backbone_h = BackboneHandler(self.backbone, num_classes, output_func=self.logistic, loss_type=self.loss_type)
        x = backbone_h.build(input_layer)

        self.model = Model(inputs=input_layer, outputs=x)
        # print(self.model.summary())

        self.load()
        if compile_model:
            # opt = keras.optimizers.Adam(learning_rate=0.00001)
            self.model.compile(loss=backbone_h.loss(), metrics=backbone_h.metric(), optimizer=self.optimizer)

    def load(self):
        model_path = None
        if os.path.isdir(self.model_folder):
            pot_models = sorted(os.listdir(self.model_folder))
            for model in pot_models:
                if model.lower().endswith((".hdf5", ".h5")):
                    model_path = os.path.join(self.model_folder, model)
            if model_path is not None:
                print("Model-Weights are loaded from: {}".format(model_path))
                self.model.load_weights(model_path, by_name=True)

        else:
            print("No Weights were found")

    def fit(self, tag_set_train, tag_set_test):
        if None in self.input_shape:
            print("Only Batch Size of 1 is possible.")
            self.batch_size = 1

        self.batch_size = np.min([len(tag_set_train), len(tag_set_test), self.batch_size])

        training_generator = DataGenerator(
            tag_set_train,
            image_size=self.input_shape,
            label_size=self.input_shape,
            batch_size=self.batch_size,
            augmentor=Augmentor(),
            label_prep=self.label_prep,
        )
        validation_generator = DataGenerator(
            tag_set_test,
            image_size=self.input_shape,
            label_size=self.input_shape,
            batch_size=self.batch_size,
            label_prep=self.label_prep
        )

        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder, "weights-final.hdf5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )

        patience = 100
        reduce_lr = ReduceLROnPlateau(factor=0.5, verbose=1, patience=int(patience*0.5))
        early_stop = EarlyStopping(monitor="val_loss", patience=patience, verbose=1)

        callback_list = [checkpoint, reduce_lr, early_stop]

        history = self.model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=callback_list,
            epochs=self.epochs,
        )
        with open(os.path.join(self.model_folder, "training_history.pkl"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)



