import os, sys, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
dist = distutils.core.run_setup("./detectron2/setup.py")

import pickle
from utils import seed_everything_custom
from data.utils import save_annotation_info
from models.models import SegmentationModel, PatchCoreModel
from models.utils import save_object_images, save_segmentation_results, get_patchcore_image_size

class Trainer:
    def __init__(self, configs, parse_annotation_file=True):
        self.configs = configs
        self.pass_valid = not os.path.isdir(os.path.join(self.configs.DATA_ROOT, self.configs.VALID_PREFIX)) or len(os.listdir(os.path.join(self.configs.DATA_ROOT, self.configs.VALID_PREFIX))) <= 1
        # setup dataset
        if parse_annotation_file:
            for prefix in [self.configs.TRAIN_PREFIX, self.configs.VALID_PREFIX]:
                if prefix=="valid" and self.pass_valid:
                    continue
                save_annotation_info(os.path.join(self.configs.DATA_ROOT, prefix), self.configs)

        self.configs.update()
        
    def run(self):
        # # =================STEP 1 : InstancceSegmentation===============
        # # Train Segmentation model
        segmentation_model = SegmentationModel(self.configs)
        segmentation_model.fit()
        segmentation_model.load()
        
        for prefix in [self.configs.TRAIN_PREFIX, self.configs.VALID_PREFIX]:
            if prefix=="valid" and self.pass_valid:
                continue
            # save train, val's segmentation outputs
            save_segmentation_results(segmentation_model, self.configs, prefix)
            # save segmented images
            save_object_images(self.configs, prefix)
        del segmentation_model

        # =================STEP 2 : Train AnomalyDetectionModels===============
        for train_path, valid_path, model_path in zip(
            self.configs.PATCHCORE_TRAIN_IMAGE_PATHS.values(),
            self.configs.PATCHCORE_VALID_IMAGE_PATHS.values(),
            self.configs.PATCHCORE_MODEL_PATHS.values(),
        ):

            image_size = int(get_patchcore_image_size(train_path))
            best_accuracy = 0
            for dimension in [512, 1024, 2048, 4096]:
                if self.pass_valid:
                    valid_path = None
                    print("No valid data...")
                    if dimension != 1024:
                        continue
                    
                print(f"""\n
                ====dimension:{dimension} process size:{image_size}====
                \n
                """)
                
                # model
                model = PatchCoreModel(
                    device = self.configs.DEVICE,
                    backbone_name = "wideresnet50",
                    flatten_dimension = dimension,
                    out_dimension = dimension,
                    image_size = image_size,
                    th_option = self.configs.ANOMALY_DETECTION_THRESHOLD_RULE
                )

                valid_result = model.fit(train_path, valid_path)
                if valid_result is None:
                    break
                acc = valid_result["accuracy"]
                print(f"Valid data accuracy:{acc}")
                if acc >= best_accuracy:
                    best_accuracy = acc
                    model.save(model_path)
                    with open(os.path.join(valid_path, "patchcore_result.pkl"), "wb") as f:
                        pickle.dump(valid_result, f)
                else:
                    break
        #             pass
                