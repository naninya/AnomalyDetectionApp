import os
import re
import numpy as np
import cv2
import json
import pickle
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

def get_keys(anot_path):
    with open(anot_path) as f:
        json_dict = json.load(f)
    
        info = json_dict["_via_img_metadata"]
    
        dataset_dicts=[]
        for index, anot_info in info.items():
    
            regions = anot_info["regions"]
            keys = {}
            for region in regions:
                for key, val in region["region_attributes"].items():
                    keys[key] = 1
            break
    return sorted(keys.keys())

def register_metadata(configs):
    for d in [configs.TRAIN_PREFIX]:
        base_dir = os.path.join(configs.DATA_ROOT, d)
        try:
            DatasetCatalog.register(d, lambda d=d: get_dataset_dict(base_dir, configs=configs))
            MetadataCatalog.get(d).set(thing_classes=configs.CLASS_NAMES)
        except Exception as e:
            # print(e)
            pass

def get_object_class_num(file_name, region, object_class_map):
    # no multiclass object
    error_file_names = []
    for key, val in region["region_attributes"].items():
        val = re.sub(r"\n", "", val)
        if len(val) > 0:
            if val not in ["", "0", "1"]:
                raise Exception(f"value must be 0 or 1, check file:{file_name}")
            if val == "":
                # background
                mask_value = 0
            elif val == "0":
                # normal object
                mask_value = 1
            else:
                # anomaly object
                mask_value = -1
            
            return object_class_map[key], mask_value
    else:
        print(region["region_attributes"])
        return None

def check_annotation_format(annot_file_path):
    with open(annot_file_path) as f:
        json_dict = json.load(f)
        error_file_names = []

        info = json_dict["_via_img_metadata"]
        dataset_dicts=[]
        for index, anot_info in info.items():
            record = {}
            file_name = anot_info["filename"]
            for region in anot_info["regions"]:
                for key, val in region["region_attributes"].items():
                    val = re.sub(r"\n", "", val)
                    # print(val)
                    if val not in ["", "0", "1"] or len("".join(region["region_attributes"].values())) == 0:
                        # print(f"value must be 0 or 1, check file:{file_name}")
                        error_file_names.append(file_name)
        if len(error_file_names) > 0:
            file_names = " , ".join(error_file_names)
            raise Exception(f"value must be 0 or 1, check file list :{file_names} ")

def update_area_configs(base_dir, configs):
    anot_path = os.path.join(base_dir, "annotations.json")

    # check_annotation_format(anot_path)

    image_dir = os.path.join(base_dir, "images")
    class_area = {}
    with open(anot_path) as f:
        json_dict = json.load(f)

        info = json_dict["_via_img_metadata"]

        dataset_dicts=[]
        for index, anot_info in info.items():
            record = {}
            file_name = anot_info["filename"]
            file_name = os.path.join(image_dir, file_name)
            try:
                height, width = cv2.imread(file_name).shape[:2]
            except Exception as e:
                print("file is not match with anot file name")

            record["file_name"] = file_name
            record["image_id"] = index
            record["height"] = height
            record["width"] = width

            regions = anot_info["regions"]
            mask = np.zeros((height, width, len(configs.CLASS_NAMES)))
            for region in regions:

                object_class, object_num = get_object_class_num(file_name, region, configs.OBJECT_CLASS_MAP)
                anno = region["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]

                # for save multiclass mask
                contours = []
                for x, y in zip(px, py):
                    contours.append([x,y])
                contours = np.array(contours)

                # マスクが被らないことを仮定している
                object_mask = cv2.fillPoly(np.zeros((height, width)), pts=[contours], color=1)
                
                # all object mask area
                if object_class not in class_area:
                    class_area[object_class] = []
                class_area[object_class].append(object_mask.sum())


        # update thres hold
        configs.CLASS_AREA = class_area
        for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
            area_mean = np.array(class_area[class_index]).mean()
            if area_mean < 2000:
                configs.TRAIN_SEG_THS[class_name] = 0.6
                configs.TEST_SEG_THS[class_name] = 0.5
            if area_mean < 500:
                configs.TRAIN_SEG_THS[class_name] = 0.5
                configs.TEST_SEG_THS[class_name] = 0.3

def save_annotation_mask(base_dir, configs):
    anot_path = os.path.join(base_dir, "annotations.json")

    # check_annotation_format(anot_path)

    image_dir = os.path.join(base_dir, "images")
    with open(anot_path) as f:
        json_dict = json.load(f)

        info = json_dict["_via_img_metadata"]

        dataset_dicts=[]
        for index, anot_info in info.items():
            record = {}
            file_name = anot_info["filename"]
            file_name = os.path.join(image_dir, file_name)
            try:
                height, width = cv2.imread(file_name).shape[:2]
            except Exception as e:
                print("file is not match with anot file name")

            record["file_name"] = file_name
            record["image_id"] = index
            record["height"] = height
            record["width"] = width

            regions = anot_info["regions"]
            mask = np.zeros((height, width, len(configs.CLASS_NAMES)))
            masks = {}
            for region in regions:
                object_class, object_num = get_object_class_num(file_name, region, configs.OBJECT_CLASS_MAP)
                anno = region["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]

                # for save multiclass mask
                contours = []
                for x, y in zip(px, py):
                    contours.append([x,y])
                contours = np.array(contours)
    #             print(file_name, mask.shape, object_class)

                # マスクが被らないことを仮定している
                object_mask = cv2.fillPoly(np.zeros((height, width)), pts=[contours], color=1)
                mask[:,:,object_class] += (object_mask*object_num).astype(np.uint8)


            # save mask
            target_dir = os.path.join(base_dir, "annotations")
            file_id = anot_info["filename"].split(".")[-2]
            mask_path = f"{os.path.join(target_dir, file_id)}.npy"
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            # print(f"mask_path:{mask_path}, {mask.astype(np.uint8).shape}")
            np.save(mask_path, mask.astype(np.int8))

def save_annotation_info(base_dir, configs):
    # Load json from via
    # via save option :
    # 1. Save region and file annotations (i.e. manual annotations)	
    # 2. Save region and file attributes. 
    # 3. Save VIA application settings. 

    anot_path = os.path.join(base_dir, "annotations.json")

    # check_annotation_format(anot_path)

    image_dir = os.path.join(base_dir, "images")
    class_area = {}
    with open(anot_path) as f:
        json_dict = json.load(f)

        info = json_dict["_via_img_metadata"]

        dataset_dicts=[]
        for index, anot_info in info.items():
            record = {}
            file_name = anot_info["filename"]
            file_name = os.path.join(image_dir, file_name)
            try:
                height, width = cv2.imread(file_name).shape[:2]
            except Exception as e:
                print(f"file is not match with anot file name: {file_name}")
                continue

            record["file_name"] = file_name
            record["image_id"] = index
            record["height"] = height
            record["width"] = width

            regions = anot_info["regions"]
            objs = []
            mask = np.zeros((height, width, len(configs.CLASS_NAMES)))
            masks = {}
            for region in regions:

                object_class, object_num = get_object_class_num(file_name, region, configs.OBJECT_CLASS_MAP)
                anno = region["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                # for save multiclass mask
                
                contours = []
                for x, y in zip(px, py):
                    contours.append([x,y])
                contours = np.array(contours)
    #             print(file_name, mask.shape, object_class)

                # マスクが被らないことを仮定している
                if mask[:,:,object_class].sum() == 0:
                    masks[object_class] = np.zeros((height, width))
                object_mask = cv2.fillPoly(np.zeros((height, width)), pts=[contours], color=1)
                mask[:,:,object_class] += (object_mask*object_num).astype(np.uint8)
                
                # all object mask area
                if object_class not in class_area:
                    class_area[object_class] = []
                class_area[object_class].append(object_mask.sum())

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": object_class,
                }
                objs.append(obj)

            # save mask
            target_dir = os.path.join(base_dir, "annotations")
            file_id = anot_info["filename"].split(".")[-2]
            mask_path = f"{os.path.join(target_dir, file_id)}.npy"
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            # print(f"mask_path:{mask_path}, {mask.astype(np.uint8).shape}")
            np.save(mask_path, mask.astype(np.int8))
            
            record["annotations"] = objs
            dataset_dicts.append(record)

    os.makedirs(os.path.join(base_dir, "tmp"), exist_ok=True)
    os.makedirs(configs.MODEL_PATH, exist_ok=True)
    with open(os.path.join(base_dir, "tmp", "dataset_dicts.pkl"), "wb") as f:
        pickle.dump(dataset_dicts, f)
    if configs.TRAIN_PREFIX in base_dir:
        with open(os.path.join(configs.MODEL_PATH, "class_area.pkl"), "wb") as f:
            pickle.dump(class_area, f) 
def get_dataset_dict(base_dir, configs):
    with open(os.path.join(base_dir, "tmp", "dataset_dicts.pkl"), "rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts