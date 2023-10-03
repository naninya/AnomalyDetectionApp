## data
from glob import glob
import os, cv2
import numpy as np
from tqdm import tqdm
def check_image_format(configs):
    for prefix in tqdm([configs.TRAIN_PREFIX, configs.VALID_PREFIX]):
    
        image_paths = sorted(glob(f"{configs.DATA_ROOT}/{prefix}/images/*.*"))
        annot_paths = sorted(glob(f"{configs.DATA_ROOT}/{prefix}/annotations/*.*"))
        try:
            assert len(image_paths) == len(annot_paths)
        except:
            raise Exception(f"{prefix}のイメージとアノテーションファイル数が違います。アノテーションしたファイルの画像と準備した画像数を確認してください")
        for image_path, annot_path in zip(
            image_paths,
            annot_paths
        ):
            image = cv2.imread(image_path)
            annotations = np.load(annot_path)
            image_id = os.path.basename(image_path).split(".")[0]
            try:
                assert image.shape[:2] == annotations.shape[:2]
                assert image.shape[-1] == 3
                assert annotations.shape[-1] == len(configs.CLASS_NAMES)
            except:
                raise Exception(f"{image_id}のアノテーションと画像を確認してください")
    return True

# check_image_format(configs)