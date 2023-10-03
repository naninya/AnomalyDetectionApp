import os, pickle
from models.models import PatchCoreModel

def check_anomaly_detection(configs):
    for class_name, path in configs.PATCHCORE_VALID_IMAGE_PATHS.items():
        with open(os.path.join(path, "patchcore_result.pkl"), "rb") as f:
            valid_result = pickle.load(f)
            pred = valid_result["anomaly_scores"] > valid_result["th"]
            label = valid_result["anomaly_labels"]
            accuracy = (pred == label).mean()
            if accuracy < 0.9:
                raise Exception(
                    f"""
                        異常検知モデルが上手く生成されていません。正常画像が揃えているか目視で確認してください。：{configs.PATCHCORE_TRAIN_IMAGE_PATHS}。\n
                        揃えていない場合はセグメンテーションを見直してください"""
                )
            error_index = pred != label
            error_files = valid_result["image_paths"][error_index]
            if len(error_files) > 0:
                error_files_str = " , ".join(error_files)
                if configs.ANOMALY_DETECTION_THRESHOLD_RULE == "no-miss":
                    print(f"精度向上のためのおすすめ-->エラーファイルの傾向を確認して、元の画像の撮像を調整してください。過検出した画像：{error_files_str}")
                else:
                    print(f"精度向上のためのおすすめ-->エラーファイルの傾向を確認して、元の画像の撮像を調整してください。見逃した画像：{error_files_str}")
    return True
# check_anomaly_detection(configs)