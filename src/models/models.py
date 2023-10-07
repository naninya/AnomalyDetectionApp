import torch
# import some common libraries
import numpy as np
import pickle
import os
import abc
import torchvision.models as models
import faiss
import timm
import torch.nn.functional as F
import cv2
from typing import Union
from torch.utils.data import DataLoader

# import some common detectron2 utilities
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.structures import Instances, ROIMasks

from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from torch.nn import functional as F
import albumentations
import albumentations.pytorch
from detectron2.engine import DefaultTrainer
from .utils import squeeze_black, get_valid_masks
from data.datasets import AnomalyDataset
from data.utils import register_metadata


PATCHCORE_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}
class AnomalyDetectors:
    def __init__(self, configs):
        self.configs = configs
        self.segmentation_predictor = SegmentationModel(self.configs)
        self.count_models = None
        self.detectors = None
        self.build()

    def build(self):
        with open(f"{self.configs.MODEL_PATH}/count_model.pkl", "rb") as f:
            self.count_models = pickle.load(f)
        self.detectors = self.generate_anomaly_detectors()

        # trainsformer
        self.transformer = albumentations.Compose([
            albumentations.Resize(self.configs.PATCHCORE_IMAGE_SIZE, self.configs.PATCHCORE_IMAGE_SIZE),
            albumentations.pytorch.transforms.ToTensorV2(),
        ])
        
        self.segmentation_predictor.load()

        self.patchcore_models = {}
        for class_name, option in self.configs.PATCHCORE_OPTIONS.items():
            if option:
                model_weight_path = self.configs.PATCHCORE_MODEL_PATHS[class_name]
                model = PatchCoreModel(
                    device = self.configs.DEVICE,
                    backbone_name = "wideresnet50",
                    flatten_dimension = self.configs.PATCHCORE_DIMENSION,
                    out_dimension = self.configs.PATCHCORE_DIMENSION,
                    image_size = self.configs.PATCHCORE_IMAGE_SIZE,
                    th_option = self.configs.ANOMALY_DETECTION_THRESHOLD_RULE
                )
                
                model.load(
                    save_dir = model_weight_path, 
                    device=self.configs.DEVICE, 
                    backbone_name="wideresnet50", 
                    th_option = self.configs.ANOMALY_DETECTION_THRESHOLD_RULE
                )
            else:
                model = None
            self.patchcore_models[class_name] = model

    def get_segmentation_results(self, original_image, th):
        results = {}
        outputs = self.segmentation_predictor.inference(original_image, th) 
        pred_masks = outputs.pred_masks.cpu().detach().numpy() > 0.5
        pred_classes = outputs.pred_classes.cpu().detach().numpy()
        for class_name, class_index in self.configs.OBJECT_CLASS_MAP.items():
            indices = np.where(pred_classes == class_index)[0]
            valid_index_masks = get_valid_masks(self.configs.CLASS_AREA[class_index], pred_masks[indices])
            results[class_name] = valid_index_masks
        return results
    
    def make_detectors(self, class_name, patchcore_options, target_object_num):
        use_patchcore = patchcore_options[class_name]
        
        def detector(original_image, valid_index_masks, return_debug_info=False):
            return_val = None
            if return_debug_info:
                mask = valid_index_masks[class_name].sum(axis=0)
                filtered_image = np.where(np.stack([mask]*3, -1)>0, original_image, 0)
                x_min, x_max, y_min, y_max = squeeze_black(filtered_image)
                crop_part_image = filtered_image[y_min:y_max, x_min:x_max]
            
            pred_count = len(valid_index_masks[class_name]) < target_object_num
            # print(class_name)
            # print(f"mask num:{len(valid_index_masks)}, class_masknum:{len(valid_index_masks[class_name])}")
            object_scores = []
            if use_patchcore:
                # valid_index_masks_for_anomaly_detection = self.get_segmentation_results(original_image, anomaly_detection_mask_th)
                for object_index, object_mask in enumerate(valid_index_masks[class_name]):
                    masked_object_image = np.where(np.stack([object_mask]*3, -1)>0, original_image, 0)
                    x_min, x_max, y_min, y_max = squeeze_black(np.where(np.stack([object_mask]*3, -1)>0, 1, 0))
                    crop_object_image = masked_object_image[y_min:y_max, x_min:x_max]
                    score, score_map = self.patchcore_models[class_name].inference_image(crop_object_image[:,:,::-1])
                    object_scores.append(score)
                if self.patchcore_models[class_name].anomaly_score_th is not None:
                    pred_anomaly = np.sort(np.array(object_scores))[:int(target_object_num)].max() > self.patchcore_models[class_name].anomaly_score_th
                else:
                    pred_anomaly = None
            else:
                pred_anomaly = False
            if return_debug_info:
                return_val = {
                    "pred":pred_count or pred_anomaly,
                    "crop_part_image":crop_part_image,
                    "object_scores":object_scores
                }
            else:
                return_val = {
                    "pred":pred_count or pred_anomaly,
                    "crop_part_image":None,
                    "object_scores":None
                }
            return return_val
        return detector

    def generate_anomaly_detectors(self):
        detectors = {}
        for class_name, class_index in self.configs.OBJECT_CLASS_MAP.items():
            detectors[class_name] = self.make_detectors(class_name, self.configs.PATCHCORE_OPTIONS, self.count_models[class_name])
        return detectors

    def run(self, original_image_path, return_debug_info=False):
        
        result = {}
        original_image = cv2.imread(original_image_path)
        for class_name, class_index in self.configs.OBJECT_CLASS_MAP.items():
            valid_index_masks = self.get_segmentation_results(original_image, self.configs.TEST_SEG_THS[class_name])
            detector_output = self.detectors[class_name](original_image, valid_index_masks, return_debug_info)
            if self.configs.VALID_ANOMALY_LABELS is None:
                anomaly_label = None
            else:
                image_id = os.path.basename(original_image_path).split(".")[0]
                anomaly_label= image_id in self.configs.VALID_ANOMALY_LABELS[class_name]

            result[class_name] = {
                    "pred" : detector_output["pred"],
                    "label" : anomaly_label,
                    "part_image" : detector_output["crop_part_image"],
                    "scores" : detector_output["object_scores"]
                }
            
        return result


class SegmentationModel:
    def __init__(self, configs):
        self.configs = configs
        try:
            register_metadata(configs)
        except Exception as e:
            # print(e)
            raise Exception(e)
        self.trainer = DefaultTrainer(self.configs.DETECTRON_CONFIGS)
        self.predictor = None

    def fit(self):
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        self.load()
    # def save(self):
    #     pass
    
    def load(self):
        self.configs.DETECTRON_CONFIGS.MODEL.WEIGHTS = os.path.join(self.configs.DETECTRON_CONFIGS.OUTPUT_DIR, "model_final.pth")
        self.predictor = CustomPredictor(self.configs.DETECTRON_CONFIGS)
    
    def inference(self, image, th=None):
        if th is not None:
            return self.predictor(image, th)
        return self.predictor(image, 0.5)


class PatchCoreModel(torch.nn.Module):
    def __init__(self, device, backbone_name, flatten_dimension, out_dimension, image_size, th_option):
        super(PatchCoreModel, self).__init__()
        self.device = device
        self.dimension = flatten_dimension
        self.image_size = image_size
        # trainsformer
        self.transformer = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.pytorch.transforms.ToTensorV2(),
        ])
        self.feature_extractor = PatchFeatureExtractor(
            device=device, 
            backbone_name=backbone_name, 
            flatten_dimension=flatten_dimension, 
            out_dimension=out_dimension
        )
        on_gpu = device == "cuda"
        self.faiss_nn = FaissNN(on_gpu=on_gpu, num_workers=8)
        self.anomaly_score_th = None
        self.th_option = th_option
        self.eval()
        self.to(device)
        
    def forward(self, batch_images):
        with torch.no_grad():
            torch.cuda.empty_cache()
            batch_images = batch_images.to("cuda").to(torch.float32)
            batchsize = batch_images.shape[0]
            # print(batch_images.shape)
            features, scales = self.feature_extractor(batch_images)
            features = np.asarray(features.detach().cpu().numpy())
            query_distances, query_nns = self.faiss_nn.run(1, features)

            # unpatch : check for image score 
            patch_scores = image_scores = query_distances.reshape(batchsize, -1, *query_distances.shape[1:]).copy()
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = image_scores.max(axis=1).flatten()
            
            # for check patch image
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to("cuda")
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=256, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy() 
        return image_scores, patch_scores
        
    def fit(self, train_path, valid_path=None):
        train_dataset = AnomalyDataset(train_path, self.transformer)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=False)
        # Extracting features with patch format
        with torch.no_grad():
            torch.cuda.empty_cache()
            out_features = []
            for batch in tqdm(train_dataloader):
                batch_image_paths = batch["image_path"]
                batch_images = batch["image"]
                batch_labels = batch["label"]
                
                batch_images = batch_images.to("cuda").to(torch.float32)
                batch_labels = batch_labels.to("cuda")
                out, ref_num_patches =self.feature_extractor(batch_images)
                out = out.detach().cpu().numpy() 
                out_features.append(out)
            features = np.concatenate(out_features, axis=0)
            print(f"exteacted all feature shape: {features.shape}")
            # Sampling features
            features = self.feature_extractor.sampler.run(features)
            print(f"sampled feature shape: {features.shape}")
            # train features
            self.faiss_nn.fit(features)
        return self.update_threshold(th=None, train_path=train_path, valid_path=valid_path, option=self.th_option)
    
    def inference_image(self, image):
        image = image / 255
        sample = dict(
            image=image
        )
        if self.transformer is not None:
            sample = self.transformer(**sample)
        inputs = sample["image"].to(self.device).to(torch.float32).unsqueeze(0)
        image_score, patch_score = self.forward(inputs)
        return image_score, patch_score
        
    def inference(self, test_path):
        test_dataset = AnomalyDataset(test_path, self.transformer)
        dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=False)

        pred_image_scores = []
        pred_masks = []
        anomaly_labels = []
        image_paths = []
        for i, batch in tqdm(enumerate(dataloader)):
            batch_image_paths = batch["image_path"]
            inputs = batch["image"]
            labels = batch["label"]

            inputs = inputs.to(self.device).to(torch.float32)
            # pred
            image_score, patch_score = self.forward(inputs)
            # images scores
            pred_image_scores.append(image_score)
            # pred masks
            pred_masks.append(patch_score)
            # labels
            anomaly_labels.append(labels)
            # image paths
            image_paths.append(np.array(batch_image_paths))
        pred_image_scores = np.concatenate(pred_image_scores)
        anomaly_labels = np.concatenate(anomaly_labels)
        pred_masks = np.concatenate(pred_masks, axis=0)
        image_paths = np.concatenate(image_paths).flatten()
        if self.anomaly_score_th is not None:
            pred_labels = pred_image_scores > self.anomaly_score_th
        else:
            pred_labels = None

        accuracy = (anomaly_labels == pred_labels).mean()
        return {
            "anomaly_scores":pred_image_scores,
            "anomaly_labels":anomaly_labels.astype(np.bool_),
            "pred_masks":pred_masks,
            "image_paths":image_paths,
            "pred_labels":pred_labels,
            "accuracy":accuracy,
            "th":self.anomaly_score_th
        }
    
    def update_threshold(self, th, train_path=None, valid_path=None, option="no-miss"):
        print(f"Valid path:{valid_path}")
        if th is not None:
            return None
        if valid_path is None:
            train_result = self.inference(train_path)
            self.anomaly_score_th = train_result["anomaly_scores"].max() * 2
            print("No valid images, threshold maybe not be correct value")
            return None
        test_result = self.inference(valid_path)
        if option == "no-miss":
            self.anomaly_score_th = test_result["anomaly_scores"][test_result["anomaly_labels"] == True].min() - 0.01
        elif option == "no-overdetection":
            self.anomaly_score_th = test_result["anomaly_scores"][test_result["anomaly_labels"] == False].max() + 0.01
        print(f"thresh hold value:{self.anomaly_score_th}")
        test_result["th"] = self.anomaly_score_th
        test_result["accuracy"] = ((test_result["anomaly_scores"] > self.anomaly_score_th) == test_result["anomaly_labels"]).mean()
        
        # with open(os.path.join(valid_path, "patchcore_result.pkl"), "wb") as f:
        #     pickle.dump(test_result, f)
        return test_result
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.faiss_nn.save(os.path.join(save_dir, "faiss_weight"))
        with open(os.path.join(save_dir, "info.pkl"), "wb") as f:
            info = {
                "anomaly_score_th":self.anomaly_score_th,
                "dimension":self.dimension,
                "image_size":self.image_size
            }
            pickle.dump(info, f)

        
    def load(self, save_dir, device, backbone_name, th_option):
        th = self.anomaly_score_th
        with open(os.path.join(save_dir, "info.pkl"), "rb") as f:
            info = pickle.load(f)
            self.__init__(device, backbone_name, info["dimension"], info["dimension"], info["image_size"], th_option)
            self.faiss_nn.load(os.path.join(save_dir, "faiss_weight"))            
            self.anomaly_score_th = info["anomaly_score_th"]
        



def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()

def post_process(results, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
    post_processed_results = []
    for results_per_image, input_per_image, image_size in zip(
        results, batched_inputs, image_sizes
    ):
        output_height = input_per_image.get("height", image_size[0])
        output_width = input_per_image.get("width", image_size[1])
        boxes = results_per_image.pred_boxes.tensor
        masks = results_per_image.pred_masks
        # detector_postprocess
        if isinstance(output_width, torch.Tensor):
            output_width_tmp = output_width.float()
            output_height_tmp = output_height.float()
            new_size = torch.stack([output_height, output_width])
        else:
            new_size = (output_height, output_width)
            output_width_tmp = output_width
            output_height_tmp = output_height

        scale_x, scale_y = (
            output_width_tmp / results_per_image.image_size[1],
            output_height_tmp / results_per_image.image_size[0],
        )
        results_per_image = Instances(new_size, **results_per_image.get_fields())

        if results_per_image.has("pred_boxes"):
            output_boxes = results_per_image.pred_boxes
        elif results_per_image.has("proposal_boxes"):
            output_boxes = results_per_image.proposal_boxes
        else:
            output_boxes = None
        assert output_boxes is not None, "Predictions must contain boxes!"

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results_per_image.image_size)

        results_per_image = results_per_image[output_boxes.nonempty()]

        # to_bitmasks
        masks = ROIMasks(results_per_image.pred_masks[:, 0, :, :]).tensor
        boxes = results_per_image.pred_boxes.tensor

        # paste_masks_in_image
        assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
        N = len(masks)
        if not isinstance(boxes, torch.Tensor):
            boxes = boxes.tensor
        device = boxes.device
        assert len(boxes) == N, boxes.shape

        img_h, img_w = output_height, output_width

        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == "cpu" or torch.jit.is_scripting():
            # CPU is most efficient when they are pasted one by one with skip_empty=True
            # so that it performs minimal number of operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks, but may have memory issue
            # int(img_h) because shape may be tensors in tracing
            BYTES_PER_FLOAT = 4
            GPU_MEM_LIMIT = 512**3
            num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (
                num_chunks <= N
            ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        img_masks = torch.zeros(
            N, img_h, img_w, device=device
        )
        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
            )

            img_masks[(inds,) + spatial_inds] = masks_chunk
        results_per_image.pred_masks = img_masks
        post_processed_results.append(results_per_image)
    return post_processed_results

class CustomPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, th):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}



            images = self.model.preprocess_image([inputs])
            features = self.model.backbone(images.tensor)

            # if detected_instances is None:
            if self.model.proposal_generator is not None:
                proposals, _ = self.model.proposal_generator(images, features, None)
            
            self.model.roi_heads.box_predictor.test_score_thresh = th
            results, _ = self.model.roi_heads(images, features, proposals, None)
            predictions = post_process(results, [inputs], images.image_sizes)[0]
            return predictions

class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
    



class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

class PatchFeatureExtractor(torch.nn.Module):
    def __init__(
        self, 
        device="cuda", 
        backbone_name="wideresnet50", 
        flatten_dimension=1024,
        out_dimension = 1024,
        patchsize=3,
        patchstride=1,
        sampler = ApproximateGreedyCoresetSampler(0.2, "cuda"),
    ):
        super(PatchFeatureExtractor, self).__init__()
        self.backbone = self._load_backbone(backbone_name)
        self.outputs = {}
        self.device = device
        self.flatten_dimension = flatten_dimension
        self.out_dimension = out_dimension
        self.patchsize = patchsize
        self.patchstride = patchstride
        self.padding = int((patchsize - 1) / 2)
        self.sampler = sampler
        self.to(device)
        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        for extract_layer, forward_hook in zip(["layer2", "layer3"], [self._forward_hook_layer2, self._forward_hook_layer3]):

            network_layer = self.backbone.__dict__["_modules"][extract_layer]
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.eval()
                
    def _load_backbone(self, backbone_name):
        return eval(PATCHCORE_BACKBONES[backbone_name])
    def _forward_hook_layer2(self, module, input, output):
        self.outputs["layer2"] = output
    def _forward_hook_layer3(self, module, input, output):
        self.outputs["layer3"] = output
        
    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            _ = self.backbone(images)
            features = []
            for output_index, (layer_name, _features) in enumerate(self.outputs.items()):                    
                # Patchfy: N x C x H x W --> N*H*W x C x Patchsize x Patchsize
                unfolder = torch.nn.Unfold(
                    kernel_size=self.patchsize, stride=self.patchstride, padding=self.padding, dilation=1
                )
                unfolded_features = unfolder(_features)
                number_of_total_patches = []
                for s in _features.shape[-2:]:
                    n_patches = (
                        s + 2 * self.padding - 1 * (self.patchsize - 1) - 1
                    ) / self.patchstride + 1
                    number_of_total_patches.append(int(n_patches))
                unfolded_features = unfolded_features.reshape(
                    *_features.shape[:2], self.patchsize, self.patchsize, -1
                )

                _features = unfolded_features.permute(0, 4, 1, 2, 3)

                patch_dims = number_of_total_patches
                if output_index == 0:
                    ref_num_patches = patch_dims

                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                _features = _features.reshape(-1, *_features.shape[-3:])
                
                # preprocess: N*H*W x C x Patchsize x Patchsize --> N*H*W  x  flatten_dimension
                _features = _features.reshape(len(_features), 1, -1)
                _features = F.adaptive_avg_pool1d(_features, self.flatten_dimension).squeeze(1)
                features.append(_features)
                
            # aggregator: merge 2 layer features
            features = torch.stack(features, dim=1)
            
            features = features.reshape(len(features), 1, -1)
            features = F.adaptive_avg_pool1d(features, self.out_dimension)
            features = torch.squeeze(features, 1)
                
        return features, ref_num_patches

