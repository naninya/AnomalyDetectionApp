import os, sys, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
dist = distutils.core.run_setup("./detectron2/setup.py")

import pickle
import numpy as np
from tqdm import tqdm
import cv2
from glob import glob
from models.models import AnomalyDetectors

class Tester:
    def __init__(self, configs):
        self.configs = configs
        self.configs.update()

        self.detectors = AnomalyDetectors(self.configs)
        
    def run(self, dir_name, file_name="test_result"):
        paths = glob(f"{dir_name}/*.*")
        results = []
        for image_index, path in tqdm(enumerate(paths)):
            image_id = os.path.basename(path).split(".")[-2]
            result = self.detectors.run(path, return_part_image=True)
            results.append(
                {
                    "image_id":image_id,
                    "result":result
                }
            )
        with open(f"{self.configs.OUTPUT_PATH}/{file_name}.pkl", "wb") as f:
            pickle.dump(results, f)
        return results