import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pickle

def squeeze_black(mask):
    if mask.shape[-1] == 3:
        mask = mask.mean(axis=-1).astype(np.uint8)
    w = np.where(mask.sum(axis=0)>0, 1, 0)
    x_min = w.argmax()
    x_max = mask.shape[1] - np.flip(w).argmax()
    h = np.where(mask.sum(axis=1)>0, 1, 0)
    y_min = h.argmax()
    y_max = mask.shape[0] - np.flip(h).argmax()
    return (x_min, x_max, y_min, y_max)

def get_instance_object_masks(gray_image):
    mode = cv2.RETR_EXTERNAL
    src = gray_image.copy().astype(np.uint8)
    contours, hier = cv2.findContours(src, mode, cv2.CHAIN_APPROX_NONE)
#     contours, hier = cv2.findContours(src, mode, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    results = []
    while idx >= 0:
        dst = np.zeros((*gray_image.shape, 3)).astype(np.uint8)
        cv2.drawContours(dst, contours, idx, color=(255, 255, 255), thickness=cv2.FILLED)
        idx = hier[0, idx, 0]
#         if np.where(dst.mean(axis=-1) > 0, 1, 0).sum() < th_area:
#             continue
        results.append(cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY))
    return results

def get_valid_area_range(class_area):
    p_75 = np.percentile(class_area, 75)
    p_25 = np.percentile(class_area, 25)
    iqr = p_75 - p_25
    min_val = p_25 - iqr*1.8
    max_val = p_75 + iqr*1.8
    return min_val, max_val

def get_valid_masks(class_area, index_masks, target_object_num=None):
    
    if len(index_masks) == 1:
        return index_masks
    
    area = index_masks.sum(axis=-1).sum(axis=-1)
    class_area = np.array(class_area)
    # class_area = np.array(class_area)
    # anomaly_scores = abs(area - class_area.mean()) / (class_area.std())
    # valid_index = np.where((anomaly_scores < 2.5) & (area > class_area.mean()*0.3))[0]
    min_val, max_val = get_valid_area_range(class_area)
    
    valid_index = np.where((area < max_val) & (area > min_val*0.7))[0]
    index_masks = index_masks[valid_index]
    
#     index_masks = np.array([_m for _m in index_masks if _m.sum()>min_area])
    merged_mask = index_masks.sum(axis=0)
    check_area_mask = np.where(merged_mask > 1, 1, 0)
    if check_area_mask.sum() == 0:
        return index_masks
    # initialize
    target_index_masks = index_masks
    remain_masks = []
    for check_area in get_instance_object_masks(check_area_mask):
        valid_masks = []
        check_masks = []
        for mask in target_index_masks:
            if np.bitwise_and(check_area, mask).sum():
                check_masks.append(mask)
            else:
                valid_masks.append(mask)

        valid_mask = check_masks[np.array([mask.sum() for mask in check_masks]).argmin()]
        remain_masks.append(check_masks[np.array([mask.sum() for mask in check_masks]).argmax()])
        valid_masks.append(valid_mask)
        target_index_masks = valid_masks
    
    if target_object_num is not None:
        if len(valid_masks) < target_object_num:
            merged_remain_mask = np.array(remain_masks).max(axis=0)
            for mask in valid_masks:
                if np.bitwise_and(merged_remain_mask, mask).sum()*1.5  > mask.sum():
                    sub_mask = np.where(mask == False, merged_remain_mask, 0)
                    valid_masks.append(sub_mask)
                    break
        
    return np.array(valid_masks)

# def get_valid_masks(index_masks, target_object_num=None):
    
#     area = index_masks.sum(axis=-1).sum(axis=-1)
#     anomaly_scores = abs(area - area.mean()) / (area.std())
# #     print(f"areas: {area}")
# #     print(f"anomaly scores: {anomaly_scores}")
#     valid_index = np.where((anomaly_scores < 2.5) & (area > 300))[0]
#     index_masks = index_masks[valid_index]
    
# #     index_masks = np.array([_m for _m in index_masks if _m.sum()>min_area])
#     merged_mask = index_masks.sum(axis=0)
#     check_area_mask = np.where(merged_mask > 1, 1, 0)
#     if check_area_mask.sum() == 0:
#         return index_masks
#     # initialize
#     target_index_masks = index_masks
#     remain_masks = []
#     for check_area in get_instance_object_masks(check_area_mask):
#         valid_masks = []
#         check_masks = []
#         for mask in target_index_masks:
#             if np.bitwise_and(check_area, mask).sum():
#                 check_masks.append(mask)
#             else:
#                 valid_masks.append(mask)

#         valid_mask = check_masks[np.array([mask.sum() for mask in check_masks]).argmin()]
#         remain_masks.append(check_masks[np.array([mask.sum() for mask in check_masks]).argmax()])
#         valid_masks.append(valid_mask)
#         target_index_masks = valid_masks
    
#     if target_object_num is not None:
#         if len(valid_masks) < target_object_num:
#             merged_remain_mask = np.array(remain_masks).max(axis=0)
#             for mask in valid_masks:
#                 if np.bitwise_and(merged_remain_mask, mask).sum()*1.5  > mask.sum():
#                     sub_mask = np.where(mask == False, merged_remain_mask, 0)
#                     valid_masks.append(sub_mask)
#                     break
        
#     return np.array(valid_masks)


def save_object_images(configs, prefix):
    if prefix == "train":
        seg_output_path = f"{configs.DATA_ROOT}/{prefix}/tmp/seg_result.pkl"
        with open(seg_output_path, "rb") as f:
            seg_outputs = pickle.load(f)
        valid_parts_num = {}
        for index, path in enumerate(glob(f"{configs.SEGMENTATION_TRAIN_IMAGE_PATH}/*.*")):
            image = cv2.imread(path)
            image_id = os.path.basename(path).split(".")[-2]

            for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
                # outputs = model.inference(image, configs.TRAIN_SEG_THS[class_name]) 
                pred_masks, pred_classes = seg_outputs[path][class_name]

                indices = np.where(pred_classes == class_index)[0]
                index_masks = pred_masks[indices]
                mask_len = len(index_masks)
                index_masks = get_valid_masks(configs.CLASS_AREA[class_index], index_masks)
                # print(f"id:{image_id} cls:{class_name} :: {mask_len}  -> {len(index_masks)}")
                if class_index not in valid_parts_num:
                    valid_parts_num[class_name] = []
                valid_parts_num[class_name].append(mask_len)
                if configs.PATCHCORE_OPTIONS[class_name]:
                    anomaly_train_dir = f"{configs.DATA_ROOT}/part_{class_name}_anomaly_{configs.TRAIN_PREFIX}/images"
                    for object_index, object_mask in enumerate(index_masks):
                        masked_object_image = np.where(np.stack([object_mask]*3, -1)>0, image, 0)
                        x_min, x_max, y_min, y_max = squeeze_black(masked_object_image)
                        crop_object_image = masked_object_image[y_min:y_max, x_min:x_max]
            
                        # save 
                        save_path = os.path.join(anomaly_train_dir, f"{image_id}_{object_index}_{0}.png")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, crop_object_image)
        # save counting models
        count_models = {}
        for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
            nums = valid_parts_num[class_name]
            count_model_num = np.ceil(np.mean(nums))
            count_models[class_name] = count_model_num

        os.makedirs(configs.MODEL_PATH, exist_ok=True)
        with open(f"{configs.MODEL_PATH}/count_model.pkl", "wb") as f:
            pickle.dump(count_models, f)
    elif prefix == "valid":
        seg_output_path = f"{configs.DATA_ROOT}/{prefix}/tmp/seg_result.pkl"
        with open(seg_output_path, "rb") as f:
            seg_outputs = pickle.load(f)

        for prefix in [configs.VALID_PREFIX]:
            for index, (image_path, anot_path) in enumerate(zip(
                        sorted(glob(f"{os.path.join(configs.DATA_ROOT, prefix)}/images/*.*")), 
                        sorted(glob(f"{os.path.join(configs.DATA_ROOT, prefix)}/annotations/*.*")
                    ))
                ):
                image = cv2.imread(image_path)
                annotation = np.load(anot_path)
                image_id = os.path.basename(image_path).split(".")[-2]

                for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
                    # outputs = model.inference(image, configs.TEST_SEG_THS[class_name]) 
                    pred_masks, pred_classes = seg_outputs[image_path][class_name]
                    if configs.PATCHCORE_OPTIONS[class_name]:
                        indices = np.where(pred_classes == class_index)[0]
                        index_masks = pred_masks[indices]
                        mask_len = len(index_masks)
                        index_masks = get_valid_masks(configs.CLASS_AREA[class_index], index_masks)
                        anomaly_train_dir = f"{configs.DATA_ROOT}/part_{class_name}_anomaly_{prefix}/images"
                        # print(f"id:{image_id} class_name:{class_name} :: {mask_len}  -> {len(index_masks)}")
                        for object_index, object_mask in enumerate(index_masks):
            
                            masked_object_image = np.where(np.stack([object_mask]*3, -1)>0, image, 0)
                            x_min, x_max, y_min, y_max = squeeze_black(masked_object_image)
                            crop_object_image = masked_object_image[y_min:y_max, x_min:x_max]
            
                            limited_area_mask = np.where(object_mask > 0, annotation[:,:,class_index], 0)[y_min:y_max, x_min:x_max]
                            normal_area = np.where(limited_area_mask > 0, 1, 0).sum()
                            anomaly_area = np.where(limited_area_mask < 0, 1, 0).sum()
                            anomaly = (anomaly_area > normal_area)*1
                            
                            # save 
                            save_path = os.path.join(anomaly_train_dir, f"{image_id}_{object_index}_{anomaly}.png")
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            cv2.imwrite(save_path, crop_object_image)



def save_segmentation_results(model, configs, prefix):
    # for prefix in tqdm([configs.TRAIN_PREFIX, configs.VALID_PREFIX]):    
    image_paths = sorted(glob(f"{configs.DATA_ROOT}/{prefix}/images/*.*"))
    
    seg_output_path = f"{configs.DATA_ROOT}/{prefix}/tmp/seg_result.pkl"
    # TODO clear seg_output
    os.makedirs(os.path.dirname(seg_output_path), exist_ok=True)
    seg_outputs = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)
        seg_outputs[image_path] = {}
        for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
            outputs = model.inference(image, configs.TRAIN_SEG_THS[class_name])
            
            seg_outputs[image_path][class_name] = [
                outputs.pred_masks.cpu().detach().numpy() > 0.5,
                outputs.pred_classes.cpu().detach().numpy()
            ]
    with open(f"{seg_output_path}", "wb") as f:
        pickle.dump(seg_outputs, f)


def get_patchcore_image_size(target_dir):
    image_shape = cv2.imread(glob(f"{target_dir}/images/*.*")[0]).shape
    image_size = min(np.ceil(np.array(image_shape[:2]).min() / 32) * 32, 256)
    return image_size

