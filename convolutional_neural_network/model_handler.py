import os
import numpy as np
from keras.layers import Input
from keras import optimizers

from keras.models import Model
from keras.callbacks import ModelCheckpoint


from convolutional_neural_network.backbone_handler import BackboneHandler
from convolutional_neural_network.logistics_handler import LogisticsHandler
from convolutional_neural_network.data_generator import DataGenerator

from preprocessing.augmentor import Augmentor
from preprocessing.preprocessor import Preprocessor


class ModelHandler:
    def __init__(self, model_folder, cfg):
        self.model_folder = model_folder

        self.color_coding = cfg.color_coding
        self.input_shape = cfg.input_shape
        self.backbone = cfg.backbone

        self.model = None

        self.optimizer = optimizers.adam(lr=1e-6)
        self.batch_size = 8
        self.epochs = 200

    def predict_tag(self, tag):
        return self.predict(tag.load_data())

    def predict(self, data):
        y_pred = self.inference(data)
        logistics_h = LogisticsHandler(log_type="fcn")
        return logistics_h.decode(y_pred)[0]

    def inference(self, data):
        preprocessor = Preprocessor(self.input_shape)
        img = preprocessor.apply(data)
        img = np.expand_dims(img, axis=0)
        return self.model.predict(img)

    def build(self, compile_model=True):
        input_layer = Input(shape=self.input_shape)
        backbone_h = BackboneHandler(self.backbone)
        x = backbone_h.build(input_layer, num_classes=self.color_coding)
        logistics_h = LogisticsHandler(log_type="fcn")
        x = logistics_h.attach(x)

        self.model = Model(inputs=input_layer, outputs=x)

        self.load()

        if compile_model:
            self.model.compile(loss=logistics_h.loss(), optimizer=self.optimizer)

    def load(self):
        model_path = None
        pot_models = sorted(os.listdir(self.model_folder))
        for model in pot_models:
            if model.lower().endswith((".hdf5", ".h5")):
                model_path = os.path.join(self.model_folder, model)
        print("Model-Weights are loaded from: {}".format(model_path))

        self.model.load_weights(model_path, by_name=True)

    def fit(self, tag_set_train, tag_set_test):
        if None in self.input_shape:
            print("Only Batch Size of 1 is possible.")
            self.batch_size = 1

        train_steps = int(len(tag_set_train) / self.batch_size)
        val_steps = int(len(tag_set_test) / self.batch_size)
        training_generator = DataGenerator(
            tag_set_train,
            target_image_size=self.input_shape[:2],
            batch_size=self.batch_size,
            augmentor=Augmentor()
        )
        validation_generator = DataGenerator(
            tag_set_test,
            target_image_size=self.input_shape[:2],
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




