import os
import numpy as np
import cv2


def main():
    path = "/home/fmuenke/datasets/storm_drain_segmentation/train/labels"

    for np_f in os.listdir(path):
        if not np_f.endswith(".npy"):
            continue
        print(np_f)
        mat = np.load(os.path.join(path, np_f))
        print(mat.shape)
        lbm = np.zeros((mat.shape[0], mat.shape[1]))
        for ch in range(mat.shape[2]):
            lbm[mat[:, :, ch] == 1] = (ch + 1) * 50
        cv2.imwrite(os.path.join(path, np_f[:-4] + ".png"), lbm)


if __name__ == "__main__":
    main()
