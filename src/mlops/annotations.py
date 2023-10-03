## annotations
import re
import json
import pickle
from tqdm import tqdm
def check_annotation_format(configs):
    for prefix in tqdm([configs.TRAIN_PREFIX, configs.VALID_PREFIX]):
        annot_file_path = f"{configs.DATA_ROOT}/{prefix}/annotations.json"

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
                raise Exception(f"オブジェクトの数は0か、１になっているか確認してください：{file_names} ")
    return True

# check_annotation_format(configs)