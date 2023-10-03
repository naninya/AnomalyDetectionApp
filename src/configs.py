import os, sys, pickle, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
dist = distutils.core.run_setup("./detectron2/setup.py")

import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
class Configs:
    DATA_ROOT = "../data"

    TRAIN_PREFIX = "train"
    VALID_PREFIX = "valid"
    TEST_PREFIX = "test"

    SEGMENTATION_TRAIN_IMAGE_PATH = f"{DATA_ROOT}/{TRAIN_PREFIX}/images"
    SEGMENTATION_VALID_IMAGE_PATH = f"{DATA_ROOT}/{VALID_PREFIX}/images"
    SEGMENTATION_TEST_IMAGE_PATH = f"{DATA_ROOT}/{TEST_PREFIX}/images"
    MODEL_PATH = "../results/models/model_0"
    OUTPUT_PATH = "../results/test_output"
    DEVICE = "cuda"

    IMAGE_DATA__FORMAT = "JPG"

    # anomaly detection
    PATCHCORE_IMAGE_SIZE = 64
    PATCHCORE_DIMENSION = 1024

    # hyper params for user
    # define object class map
    CLASS_NAMES = ["p1", "p2", "p3", "p4"]
    OBJECT_CLASS_MAP = {_class:i for i, _class in enumerate(CLASS_NAMES)}
    PATCHCORE_OPTIONS = {
        "p1":False,
        "p2":False,
        "p3":True,
        "p4":True
    }
    VALID_ANOMALY_LABELS = None
    ANOMALY_DETECTION_THRESHOLD_RULE = "no-miss"
    
    DETECTRON_CONFIGS = get_cfg()
    DETECTRON_CONFIGS.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    DETECTRON_CONFIGS.DATASETS.TRAIN = (TRAIN_PREFIX)
    DETECTRON_CONFIGS.DATASETS.TEST = ()
    DETECTRON_CONFIGS.DATALOADER.NUM_WORKERS = 2
    DETECTRON_CONFIGS.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    DETECTRON_CONFIGS.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
    DETECTRON_CONFIGS.SOLVER.BASE_LR = 0.00025  # pick a good LR
    DETECTRON_CONFIGS.SOLVER.MAX_ITER = 3000
    DETECTRON_CONFIGS.SOLVER.STEPS = []        # do not decay learning rate
    DETECTRON_CONFIGS.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    DETECTRON_CONFIGS.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    DETECTRON_CONFIGS.OUTPUT_DIR = MODEL_PATH

    # update target configs
    PATCHCORE_TRAIN_IMAGE_PATHS = {}
    PATCHCORE_VALID_IMAGE_PATHS = {}
    PATCHCORE_TEST_IMAGE_PATHS = {}
    PATCHCORE_MODEL_PATHS = {}
    TRAIN_SEG_THS = {}
    TEST_SEG_THS = {}
    CLASS_AREA = {}

    def update(self):
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        self.SEGMENTATION_TRAIN_IMAGE_PATH = f"{self.DATA_ROOT}/{self.TRAIN_PREFIX}/images"
        self.SEGMENTATION_VALID_IMAGE_PATH = f"{self.DATA_ROOT}/{self.VALID_PREFIX}/images"
        self.SEGMENTATION_TEST_IMAGE_PATH = f"{self.DATA_ROOT}/{self.TEST_PREFIX}/images"
        self.OBJECT_CLASS_MAP = {_class:i for i, _class in enumerate(self.CLASS_NAMES)}
        for object_name, object_index in self.OBJECT_CLASS_MAP.items():
            if self.PATCHCORE_OPTIONS[object_name]:
                self.PATCHCORE_TRAIN_IMAGE_PATHS[object_name] = os.path.join(self.DATA_ROOT, f"part_{object_name}_anomaly_{self.TRAIN_PREFIX}")
                self.PATCHCORE_VALID_IMAGE_PATHS[object_name] = os.path.join(self.DATA_ROOT, f"part_{object_name}_anomaly_{self.VALID_PREFIX}")
                self.PATCHCORE_TEST_IMAGE_PATHS[object_name] = os.path.join(self.DATA_ROOT, f"part_{object_name}_anomaly_{self.TEST_PREFIX}")
                self.PATCHCORE_MODEL_PATHS[object_name] = f"{self.MODEL_PATH}/patchcore_part_{object_name}"
        
        # update thres hold
        with open(os.path.join(self.MODEL_PATH, "class_area.pkl"), "rb") as f:
            self.CLASS_AREA = pickle.load(f)
        
        for class_name, class_index in self.OBJECT_CLASS_MAP.items():
            area_mean = np.array(self.CLASS_AREA[class_index]).mean()
            self.TRAIN_SEG_THS[class_name] = 0.9
            self.TEST_SEG_THS[class_name] = 0.7
            if area_mean < 2000:
                self.TRAIN_SEG_THS[class_name] = 0.6
                self.TEST_SEG_THS[class_name] = 0.5
            if area_mean < 500:
                self.TRAIN_SEG_THS[class_name] = 0.5
                self.TEST_SEG_THS[class_name] = 0.3
            
        