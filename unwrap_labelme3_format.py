import argparse
import os
import cv2
import shutil
import numpy as np
import xml.etree.cElementTree as ET


from semantic_segmentation.data_structure.folder import Folder


CLASS_MAPPING = {
        "top": 0,
        "edge": 1,
        "edge_lowered": 2,
        "curbstone": 3,
        "leaves": 4,
    }


def create_label_map(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # create the mask array with image dimensions
    img_size = root.find('imagesize')
    nrows = img_size.find('nrows').text
    ncols = img_size.find('ncols').text
    n_classes = len(CLASS_MAPPING)
    mask_array = np.zeros((int(nrows), int(ncols), n_classes))

    class_masks = {cls: np.zeros((int(nrows), int(ncols), 3)) for cls in CLASS_MAPPING}

    for obj in root.findall('object'):
        layer = obj.find('name').text

        if layer not in CLASS_MAPPING:
            CLASS_MAPPING[layer] = len(CLASS_MAPPING)

        polygon = obj.find('polygon')
        points = []
        for pt in polygon.findall('pt'):
            x = round(float(pt.find('x').text))
            y = round(float(pt.find('y').text))
            points.append((x, y))
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(class_masks[layer], pts=[points], color=(1, 0, 0))

    for cls in class_masks:
        class_only_mask = class_masks[cls]
        mask_array[:, :, CLASS_MAPPING[cls]] = np.max(class_only_mask, axis=2)
    return mask_array


def export_sub_folder(src_path, dst_path):
    src_path = Folder(src_path)
    dst_path = Folder(dst_path)
    dst_img_path = Folder(os.path.join(str(dst_path), "images"))
    dst_lbm_path = Folder(os.path.join(str(dst_path), "labels"))
    dst_path.check_n_make_dir()
    dst_img_path.check_n_make_dir()
    dst_lbm_path.check_n_make_dir()

    for src_f in os.listdir(str(src_path)):
        if src_f.endswith((".jpg", ".png")):
            shutil.copy(
                os.path.join(str(src_path), src_f),
                os.path.join(str(dst_img_path), src_f)
            )
        elif src_f.endswith(".xml"):
            mask = create_label_map(os.path.join(str(src_path), src_f))
            # cv2.imwrite(os.path.join(str(dst_lbm_path), src_f.replace(".xml", ".png")), mask)
            np.save(os.path.join(str(dst_lbm_path), src_f.replace(".xml", ".npy")), mask)


def main(args_):
    src_df = args_.src_data_set
    dst_df = args_.dst_data_set

    Folder(dst_df).check_n_make_dir()

    for sub_f in os.listdir(src_df):
        export_sub_folder(os.path.join(src_df, sub_f), os.path.join(dst_df, sub_f))

    s = ""
    for cls in CLASS_MAPPING:
        s += "\"{}\": {},\n".format(cls, CLASS_MAPPING[cls])
        # s += "\"{}\": [[{}, {}, {}], [{}, {}, {}]],\n".format(
        #     cls,
        #     CLASS_MAPPING[cls][2], CLASS_MAPPING[cls][1], CLASS_MAPPING[cls][0],
        #     CLASS_MAPPING[cls][2], CLASS_MAPPING[cls][1], CLASS_MAPPING[cls][0],
        # )
    with open(os.path.join(dst_df, "class_mapping.txt"), "w") as f:
        f.write(s)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_data_set",
        "-src",
        help="Path to directory with the original labels",
    )
    parser.add_argument(
        "--dst_data_set",
        "-dst",
        help="Path to export the data set to",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
