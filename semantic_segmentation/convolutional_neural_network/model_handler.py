import os
import numpy as np
from keras.layers import Input
from keras import optimizers

from keras.models import Model
from keras.callbacks import ModelCheckpoint


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

        self.model = None

        self.optimizer = optimizers.adam(lr=1e-6)
        if hasattr(cfg, "batch_size"):
            self.batch_size = cfg.opt["backbone"]
        else:
            self.batch_size = 1
        self.epochs = 200

    def predict_tag(self, tag):
        return self.predict(tag.load_data())

    def predict(self, data):
        y_pred = self.inference(data)
        logistics_h = LogisticsHandler(log_type="fcn", num_classes=len(self.color_coding))
        return logistics_h.decode(y_pred, self.color_coding)

    def inference(self, data):
        preprocessor = Preprocessor(self.input_shape)
        img = preprocessor.apply(data)
        img = np.expand_dims(img, axis=0)
        return self.model.predict(img)

    def build(self, compile_model=True):
        input_layer = Input(batch_shape=(self.batch_size,
                                         self.input_shape[0],
                                         self.input_shape[1],
                                         self.input_shape[2],
                                         ))
        backbone_h = BackboneHandler(self.backbone, len(self.color_coding))
        x = backbone_h.build(input_layer)
        logistics_h = LogisticsHandler(log_type="fcn", num_classes=len(self.color_coding))

        self.model = Model(inputs=input_layer, outputs=x)

        print(self.model.summary())

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
            image_size=self.input_shape[:2],
            label_size=self.input_shape[:2],
            batch_size=self.batch_size,
            augmentor=Augmentor()
        )
        validation_generator = DataGenerator(
            tag_set_test,
            image_size=self.input_shape[:2],
            label_size=self.input_shape[:2],
            batch_size=self.batch_size
        )

        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"),
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
            mode="max",
        )

        callback_list = [checkpoint]

        self.model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            callbacks=callback_list,
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
        )




