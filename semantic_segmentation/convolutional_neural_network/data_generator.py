from tensorflow import keras
import numpy as np

from semantic_segmentation.preprocessing.preprocessor import Preprocessor


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        tag_set,
        batch_size,
        image_size,
        label_size,
        shuffle=False,
        augmentor=None,
        padding=False,
        label_prep=None,
    ):
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.tag_set = tag_set
        self.shuffle = shuffle
        self.augmentor = augmentor
        self.padding = padding
        self.label_prep = label_prep
        self.indexes = np.arange(len(self.tag_set))
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.tag_set) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns one batch of data.
        Args:
            index (int)
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        tags_temp = [self.tag_set[k] for k in indexes]

        x, y = self.__data_generation(tags_temp)

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tags_temp):
        """
        Generates data containing the batch_size samples.
        """
        x = []
        y1 = []
        for i, tag in enumerate(tags_temp):
            img = tag.load_x()
            lab1 = tag.load_y((img.shape[0], img.shape[1]), label_prep=self.label_prep)

            if self.augmentor is not None:
                img, lab1 = self.augmentor.apply(img, lab1)

            preprocessor = Preprocessor(image_size=self.image_size)
            img = preprocessor.apply(img)
            lab1 = preprocessor.apply_to_label_map(lab1)
            x.append(img)
            y1.append(lab1)

        x = np.array(x)
        y = np.array(y1)

        return x, y
