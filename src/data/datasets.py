import os
import torch
import cv2
import numpy as np
from glob import glob
from PIL import Image
import albumentations
import albumentations.pytorch

class AnomalyDataset(torch.utils.data.Dataset): 
    def __init__(
        self,
        root_dir,
        transform=None,
        anomaly_ids=[], # id:label
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.anomaly_ids = anomaly_ids
        self._split()
    def __len__(self):
        return len(self.image_paths)
    
    def _split(self):
        for index, path in enumerate(sorted(glob(f"{self.root_dir}/images/*.*"))):
            image_id = os.path.basename(path)
            label = image_id.split(".")[0][-1]
            self.image_paths.append(path)
            self.labels.append(int(label))
                
                
    def __getitem__(self, idx): 
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))/255
        label = self.labels[idx]
        sample = dict(
            image=image,
            image_path=image_path,
            label=label,
        )
        if self.transform is not None:
            sample = self.transform(**sample)
            return sample
        return sample
