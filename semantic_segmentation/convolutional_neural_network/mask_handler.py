from tqdm import tqdm
import numpy as np

from semantic_segmentation.options.config import load_config
from semantic_segmentation.convolutional_neural_network.model_handler import ModelHandler


class MaskHandler:
    def __init__(self, model_folder, target_classes):
        cfg = load_config(model_folder)
        self.model_h = ModelHandler(model_folder, cfg)
        self.model_h.batch_size = 1
        self.model_h.build()

        self.target_classes = target_classes

    def mask_tag(self, tag):
        color_coding = self.model_h.predict(tag.load_data())
        h, w = color_coding.shape[:2]
        mask = np.zeros((h, w))
        for x in range(w):
            for y in range(h):
                for tc in self.target_classes:
                    a = color_coding[y, x] == self.model_h.color_coding[tc][1]
                    if a.all():
                        mask[y, x] = 1
        tag.set_maks(mask)

    def mask_tag_set(self, tag_set):
        print("Masking tags in set...")
        for tid in tqdm(tag_set):
            self.mask_tag(tag_set[tid])
        return tag_set
