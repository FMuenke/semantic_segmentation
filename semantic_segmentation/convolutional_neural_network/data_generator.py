import keras
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
    ):
        """
        Args:
            images_paths (list): List with the paths of images.
            labels (list): Labels of the images.
            batch_size (int): Batch size.
            resized_image_size (tuple): All images will have this size for
                training. If you don't want to resize your images just give it
                the original image size.
            n_channels (int): Channels of the images. Defaults to 3.
            shuffle (bool): If true, the data will be shuffled.
        """
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.tag_set = tag_set
        self.shuffle = shuffle
        self.augmentor = augmentor
        self.padding = padding
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
        self.indexes = np.arange(len(self.tag_set))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tags_temp):
        """
        Generates data containing the batch_size samples.
        """
        x = []
        y = []

        for i, tag in enumerate(tags_temp):
            img = tag.load_x()
            lab = tag.load_y((img.shape[0], img.shape[1]))

            if self.augmentor is not None:
                img, lab = self.augmentor.apply(img, lab)

            preprocessor = Preprocessor(image_size=self.image_size, padding=self.padding)
            img = preprocessor.apply(img)
            lab = preprocessor.apply(lab)
            x.append(img)
            y.append(lab)

        x = np.array(x)
        y = np.array(y)

        return x, y
