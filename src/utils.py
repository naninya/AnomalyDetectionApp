import random
import numpy as np
import os
import torch
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import cv2

# fix seeds
def seed_everything_custom(seed: int = 91):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False    
    seed_everything(seed)
    print(f"seed{seed}, applied")

def show_results(configs, results):
    # show results
    fig = plt.figure(figsize=(18, 8*len(results)))
    axes = fig.subplots(len(results), len(configs.CLASS_NAMES)+1)
    for image_index, result in enumerate(results):
        image_id = result["image_id"]
        original_image_path = f"{configs.DATA_ROOT}/{configs.VALID_PREFIX}/images/{image_id}.{configs.IMAGE_DATA__FORMAT}"
        image = cv2.imread(original_image_path)

        axes[image_index][0].imshow(image)
        axes[image_index][0].set_title(image_id)
        for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
            pred_anomaly = result["result"][class_name]["pred"]
            anomaly_label = result["result"][class_name]["label"]
            part_image = result["result"][class_name]["part_image"]
            axes[image_index][class_index+1].imshow(part_image)
            axes[image_index][class_index+1].set_title(f"pred:{pred_anomaly}/ ans:{anomaly_label}")
    plt.show()

def get_error_info(configs, results):
    error_image_info = {}
    for result in results:
        image_id = result["image_id"]
        for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
            pred = result["result"][class_name]["pred"]
            label = result["result"][class_name]["label"]
            
            if pred!=label:
                if image_id not in error_image_info:
                    error_image_info[image_id] = []
                error_image_info[image_id].append({
                        "erorr_parts_name":class_name,
                        "pred":pred,
                        "label":label
                    }
                )
    return error_image_info

def get_valid_accuracies(configs, results):
    class_accuracies = {}
    for result in results:
        image_id = result["image_id"]
        for class_name, class_index in configs.OBJECT_CLASS_MAP.items():
            pred = result["result"][class_name]["pred"]
            label = result["result"][class_name]["label"]
            if class_name not in class_accuracies:
                class_accuracies[class_name] = []
            class_accuracies[class_name].append(pred==label)
    class_accuracies = {class_name:np.array(vals).mean() for class_name, vals in class_accuracies.items()}
    return class_accuracies



def show_parts_accuracies(class_accuracies):
    plt.figure(figsize=(18,8))
    plt.title("Accuracy by parts")
    plt.bar(class_accuracies.keys(), class_accuracies.values())
    plt.xlabel("parts name")
    plt.ylabel("accuracy")
    plt.show()
