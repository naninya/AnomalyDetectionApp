import os, sys, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
dist = distutils.core.run_setup("./detectron2/setup.py")

from utils import seed_everything_custom
from data.utils import save_annotation_info
from models.models import SegmentationModel, PatchCoreModel
from models.utils import save_object_images, save_segmentation_results

class Trainer:
    def __init__(self, configs, parse_annotation_file=True):
        self.configs = configs
        # setup dataset
        if parse_annotation_file:
            for prefix in [self.configs.TRAIN_PREFIX, self.configs.VALID_PREFIX]:
                save_annotation_info(os.path.join(self.configs.DATA_ROOT, prefix), self.configs)

        self.configs.update()
        
    def run(self):
        # =================STEP 1 : InstancceSegmentation===============
        # Train Segmentation model
        segmentation_model = SegmentationModel(self.configs)
        segmentation_model.fit()
        segmentation_model.load()

        # save train, val's segmentation outputs
        save_segmentation_results(segmentation_model, self.configs)

        del segmentation_model
        # save segmented images
        save_object_images(self.configs)


        # =================STEP 2 : Train AnomalyDetectionModels===============
        for train_path, valid_path, model_path in zip(
            self.configs.PATCHCORE_TRAIN_IMAGE_PATHS.values(),
            self.configs.PATCHCORE_VALID_IMAGE_PATHS.values(),
            self.configs.PATCHCORE_MODEL_PATHS.values(),
        ):
            print(f"""\n
            ====dimension:{self.configs.PATCHCORE_DIMENSION}====
            \n
            """)
            
            # model
            model = PatchCoreModel(
                device = self.configs.DEVICE,
                backbone_name = "wideresnet50",
                flatten_dimension = self.configs.PATCHCORE_DIMENSION,
                out_dimension = self.configs.PATCHCORE_DIMENSION,
                image_size = self.configs.PATCHCORE_IMAGE_SIZE,
                th_option = self.configs.ANOMALY_DETECTION_THRESHOLD_RULE
            )
            model.fit(train_path, valid_path)
            model.save(model_path)
        del model