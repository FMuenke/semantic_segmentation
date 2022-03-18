import cv2
import numpy as np
import pathlib
from argparse import ArgumentParser


def get_crops(img_path, label_path, crop_size):
    crops = []
    crop_labels = []

    img = cv2.imread(str(img_path))
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]

    min_axis, min_size = np.argmin(np.array([height, width])), min(height, width)  # axis 0 is height, axis 1 is width
    max_size = max(height, width)
    max_axis = 1 if min_axis == 0 else 0  # avoid problem with squared img

    if crop_size % min_size == 0:
        if min_size > crop_size:
            crops_min_axis_possible = int(min_size / crop_size)
        else:
            crops_min_axis_possible = int(crop_size / min_size)
    else:
        crops_min_axis_possible = int(min_size / crop_size) + 1

    scale_factor = crops_min_axis_possible * crop_size / min_size
    if img is None:
        return
    if label is None:
        return
    resized_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    resized_label = cv2.resize(label, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    # crops along bigger axis
    max_size = max_size * scale_factor

    if max_size % crop_size == 0:
        crops_max_axis = int(max_size / crop_size)
    else:
        crops_max_axis = int(max_size / crop_size) + 1

    offset = crops_max_axis * crop_size - max_size

    for i in range(crops_min_axis_possible):
        for j in range(crops_max_axis):
            arr = np.zeros(shape=4).astype(int)  # height_start, height_end, width_start, width_end
            arr[min_axis*2:min_axis*2 + 2] = [crop_size * i, crop_size * (i + 1)]
            if j == 0:
                arr[(max_axis * 2):(max_axis * 2) + 2] = [0, crop_size]
            else:
                arr[(max_axis * 2):(max_axis * 2) + 2] = [crop_size * j - offset, crop_size * (j + 1) - offset]

            crop = resized_img[arr[0]:arr[1], arr[2]:arr[3], :]
            crop_label = resized_label[arr[0]:arr[1], arr[2]:arr[3]]

            crops.append(crop)
            crop_labels.append(crop_label)

    return crops, crop_labels


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", help="source directory with images and files folder")
    parser.add_argument("-t", "--target_dir", help="target directory for the cropped images/labels")
    parser.add_argument("-s", "--size", help="crop size size x size", type=int)

    args = parser.parse_args()

    img_dir = pathlib.Path("{}/images".format(args.dir))
    label_dir = pathlib.Path("{}/labels".format(args.dir))

    target_img_dir = pathlib.Path("{}/crops".format(args.target_dir))
    target_img_dir.mkdir(exist_ok=True)
    target_labels_dir = pathlib.Path("{}/crop_labels".format(args.target_dir))
    target_labels_dir.mkdir(exist_ok=True)

    for img in img_dir.iterdir():
        label = pathlib.Path("{}/{}_label.tif".format(str(label_dir), img.name[:-4]))
        crops, crop_labels = get_crops(img, label, args.size)

        for idx, (crop, crop_label) in enumerate(zip(crops, crop_labels)):
            if np.sum(crop_label) != 0:  # skip empty labels
                cv2.imwrite("{}/{}_{}.png".format(str(target_img_dir), str(img.name)[:-4], idx), crop)
                cv2.imwrite("{}/{}_{}.png".format(str(target_labels_dir), str(label.name)[:-4], idx), crop_label)
