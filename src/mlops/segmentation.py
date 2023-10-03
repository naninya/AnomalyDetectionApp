## Segmentation
import os
import torch
import pickle
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from torchmetrics import JaccardIndex
from models.models import SegmentationModel
from models.utils import get_valid_masks, get_valid_area_range, get_instance_object_masks, squeeze_black

def check_segmentation(configs):
    jaccard = JaccardIndex(task="binary", num_classes=2)
    
    image_ious = {}
    for prefix in tqdm([configs.TRAIN_PREFIX, configs.VALID_PREFIX]):


        seg_output_path = f"{configs.DATA_ROOT}/{prefix}/tmp/seg_result.pkl"
        with open(seg_output_path, "rb") as f:
            seg_outputs = pickle.load(f)
        
        image_paths = sorted(glob(f"{configs.DATA_ROOT}/{prefix}/images/*.*"))
        annot_paths = sorted(glob(f"{configs.DATA_ROOT}/{prefix}/annotations/*.*"))
        
        for image_path, annot_path in zip(
            image_paths,
            annot_paths
        ):
            image = cv2.imread(image_path)
            annotation = np.load(annot_path)
            image_id = os.path.basename(image_path).split(".")[-2]
            
            for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
                pred_masks, pred_classes = seg_outputs[image_path][class_name]
                
                indices = np.where(pred_classes == class_index)[0]
                index_masks = pred_masks[indices]
                mask_len = len(index_masks)
                index_masks = get_valid_masks(configs.CLASS_AREA[class_index], index_masks)
                index_annotation = annotation[:,:,class_index]
    
                iou = jaccard(torch.Tensor(index_masks.sum(axis=0)), torch.Tensor(abs(index_annotation)))
                if prefix not in image_ious:
                    image_ious[prefix] = []
                image_ious[prefix].append(iou)
    image_ious_mean = {key:np.mean(val) for key, val in image_ious.items()}
    if image_ious_mean["train"] < 0.7:
        raise Exception(f"セグメンテーション学習が上手く行っていません。アノテーションを再作成するか、特徴が明確な領域に変更してアノテーションをしてください")
    if image_ious_mean["valid"] < 0.7:
        raise Exception(f"セグメンテーション学習は上手く出来ましたが、検証データの特性が学習データとは違います。学習データと同じ角度や環境で撮っているかを確認してください")
    return True
# check_segmentation(configs)