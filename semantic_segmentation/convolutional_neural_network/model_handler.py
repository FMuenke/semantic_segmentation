import os
import numpy as np
import cv2
from keras.layers import Input
from keras import optimizers

from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


from semantic_segmentation.convolutional_neural_network.backbones.backbone_handler import BackboneHandler
from semantic_segmentation.convolutional_neural_network.logistics_handler import LogisticsHandler
from semantic_segmentation.convolutional_neural_network.data_generator import DataGenerator

from semantic_segmentation.preprocessing.augmentor import Augmentor
from semantic_segmentation.preprocessing.preprocessor import Preprocessor


class ModelHandler:
    def __init__(self, model_folder, cfg):
        self.model_folder = model_folder

        self.color_coding = cfg.color_coding
        self.input_shape = cfg.opt["input_shape"]
        self.backbone = cfg.opt["backbone"]
        self.padding = cfg.opt["padding"]
        if "loss" in cfg.opt:
            self.loss_type = cfg.opt["loss"]
        else:
            self.loss_type = "binary_crossentropy"

        if "label_prep" in cfg.opt:
            self.label_prep = cfg.opt["label_prep"]
        else:
            self.label_prep = "basic"
        self.model = None

        self.optimizer = optimizers.adam(lr=cfg.opt["init_learning_rate"])
        if "batch_size" in cfg.opt:
            self.batch_size = cfg.opt["batch_size"]
        else:
            self.batch_size = 1
        self.epochs = 200

    def predict_tag(self, tag):
        return self.predict(tag.load_data())

    def predict(self, data):
        y_pred = self.inference(data)
        logistics_h = LogisticsHandler(loss_type=None, num_classes=len(self.color_coding))
        return logistics_h.decode(y_pred, self.color_coding)

    def inference(self, data):
        preprocessor = Preprocessor(self.input_shape, padding=self.padding)
        img = preprocessor.apply(data)
        img = np.expand_dims(img, axis=0)
        res = self.model.predict(img)
        res = preprocessor.un_apply(res)
        return res

    def build(self, compile_model=True):
        input_layer = Input(batch_shape=(self.batch_size,
                                         self.input_shape[0],
                                         self.input_shape[1],
                                         self.input_shape[2],
                                         ))
        if self.loss_type == "mean_squared_error":
            output_func = "linear"
        else:
            output_func = "sigmoid"
        num_classes = len(self.color_coding)
        if self.label_prep == "ellipse":
            num_classes = 3
        backbone_h = BackboneHandler(self.backbone, num_classes, output_func)
        x = backbone_h.build(input_layer)
        logistics_h = LogisticsHandler(loss_type=self.loss_type, num_classes=len(self.color_coding))

        self.model = Model(inputs=input_layer, outputs=x)

        self.load()

        if compile_model:
            self.model.compile(loss=logistics_h.loss(), optimizer=self.optimizer, metrics=["accuracy"])

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
                print("Model-Weights are loaded from: {}".format(model_path))

        else:
            print("No Weights were found")

    def fit(self, tag_set_train, tag_set_test):
        if None in self.input_shape:
            print("Only Batch Size of 1 is possible.")
            self.batch_size = 1

        train_steps = int(len(tag_set_train) / self.batch_size)
        val_steps = int(len(tag_set_test) / self.batch_size)
        training_generator = DataGenerator(
            tag_set_train,
            image_size=self.input_shape,
            label_size=self.input_shape,
            batch_size=self.batch_size,
            augmentor=Augmentor(),
            padding=self.padding,
            label_prep=self.label_prep,
        )
        validation_generator = DataGenerator(
            tag_set_test,
            image_size=self.input_shape,
            label_size=self.input_shape,
            batch_size=self.batch_size,
            padding=self.padding,
            label_prep=self.label_prep
        )

        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder, "weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"),
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
            mode="max",
        )

        reduce_lr = ReduceLROnPlateau(factor=0.5)

        callback_list = [checkpoint, reduce_lr]



        self.model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            callbacks=callback_list,
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
        )




