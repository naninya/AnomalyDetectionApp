from fastapi import APIRouter, Path, HTTPException, status, Request, Depends, Form, File, UploadFile

from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict

from model import RequestManager
import shutil
import os
import re
import pickle
from trainers import Trainer
from testers import Tester
from data.utils import get_keys
from configs import Configs
from debuggers import DefaultDebugger
from glob import glob
router = APIRouter()

templates = Jinja2Templates(directory="templates/")
configs = Configs()
upload_dict = {
    "train_image_upload_num":0,
    "valid_image_upload_num":0,
    "test_image_upload_num":0,
    "train_annotation_upload":False,
    "valid_annotation_upload":False,
    "annotation_keys":[],
    "parchcore_options":[],
    "model_list":[],
    "test_list":[],
}

def update_upload_state():
    for prefix in ["train", "valid", "test"]:
        path = f"{configs.DATA_ROOT}/{prefix}/images"
        if os.path.isdir(path):
            image_num = len(os.listdir(path))
        else:
            image_num = 0
        upload_dict[f"{prefix}_image_upload_num"] = image_num
            
        path = f"{configs.DATA_ROOT}/{prefix}/annotations.json"
        if os.path.isfile(path):
            upload_dict[f"{prefix}_annotation_upload"] = True
            if prefix == "train":
                upload_dict[f"annotation_keys"] = get_keys(path)
        else:
            upload_dict[f"{prefix}_annotation_upload"] = False

    upload_dict[f"model_list"] = []
    upload_dict[f"test_list"] = []
    try:
        for path in glob("../results/models/*"):
            upload_dict[f"model_list"].append(os.path.basename(path))
        for path in glob("../results/test_output/*"):
            upload_dict[f"test_list"].append(os.path.basename(path).split(".")[-2])
    except:
        upload_dict[f"model_list"] = []
        upload_dict[f"test_list"] = []
    if len(upload_dict["parchcore_options"]) == 0:
        upload_dict["parchcore_options"] = [False for i in range(len(upload_dict))]
    elif len(glob(f"{configs.DATA_ROOT}/part_p*")) > 0:
        upload_dict["parchcore_options"] = []
        reg = re.compile(r"part_(p\d)")
        parts = []
        for path in glob(f"{configs.DATA_ROOT}/part_p*"):
            parts.append(reg.search(path).group(1))
        for key in upload_dict["annotation_keys"]:
            upload_dict["parchcore_options"].append(key in set(parts))
    print(upload_dict)
    
update_upload_state()


@router.get("/")
def home(request: Request):
    update_upload_state()
    return return_template(request)

@router.post("/show_result")
def result(
        request: Request,
        selected_result_name: str = Form(),
    ):
    
    with open(f"{configs.OUTPUT_PATH}/{selected_result_name}.pkl", "rb") as f:
        inference_result = pickle.load(f)
    outputs = {}
    print(inference_result)
    for ret in inference_result:
        preds = []
        labels = []
        for part_name, val in ret["result"].items():
            preds.append(val["pred"])
            labels.append(val["label"])
        outputs[ret["image_id"]] = {
            "preds":preds,
            "labels":labels
        }
    return templates.TemplateResponse("result.html",
    {
        "request": request,
        "outputs":outputs
    })



@router.post("/image_upload/{prefix}")
async def upload_file(
        prefix: str, 
        request: Request,
        files: List[UploadFile]
    ):
    for _file in files:
        target_path = f"{configs.DATA_ROOT}/{prefix}/images/{_file.filename}"
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            content = _file.file.read()
            f.write(content)
    return return_template(request)

@router.post("/annotation_upload/{prefix}")
async def upload_file(
        prefix: str,
        request: Request,
        file: bytes = File()
    ):
    target_path = f"{configs.DATA_ROOT}/{prefix}/annotations.json"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(file)
    return return_template(request)

@router.post("/update_configs/{key}")
async def update_configs(
        key:str,
        request: Request,
        part_value:str = Form(...),
    ):
    idx = upload_dict["annotation_keys"].index(key)
    upload_dict["parchcore_options"][idx] = part_value == "True"
    return return_template(request)

@router.post("/train")
def train(
        request: Request,
        model_name: str = Form(),
    ):
    print(f"start training:{model_name}")
    configs = Configs()
    configs.MODEL_PATH = f"../results/models/{model_name}"
    configs.CLASS_NAMES = upload_dict["annotation_keys"]
    for index, cls_name in enumerate(configs.CLASS_NAMES):
        configs.PATCHCORE_OPTIONS[cls_name] = upload_dict["parchcore_options"][index]
    trainer = Trainer(configs, parse_annotation_file=True)
    trainer.run()
    return return_template(request)
    
    
@router.post("/valid")
def valid(
        request: Request,
        selected_model_name: str=Form(),
    ):
    print(f"start validation debugging..")
    configs = Configs()
    configs.CLASS_NAMES = upload_dict["annotation_keys"]
    configs.MODEL_PATH = f"../results/models/{selected_model_name}"
    for index, cls_name in enumerate(configs.CLASS_NAMES):
        configs.PATCHCORE_OPTIONS[cls_name] = upload_dict["parchcore_options"][index]
    target_folder = configs.SEGMENTATION_VALID_IMAGE_PATH
    print(configs.PATCHCORE_OPTIONS)
    tester = Tester(configs)
    result = tester.run(
        dir_name = target_folder, 
        file_name = f"{selected_model_name}_valid_result"
    )
    debugger = DefaultDebugger(configs)
    outputs = debugger.run()
    return templates.TemplateResponse("valid_result.html",
    {
        "request": request,
        "outputs":outputs
    })
    
@router.post("/test")
def inference(
        request: Request,
        selected_model_name: str=Form(),
        test_name: str=Form(),
    ):
    print(f"start inference..")
    
    configs = Configs()
    target_folder = configs.SEGMENTATION_TEST_IMAGE_PATH
    
    configs.CLASS_NAMES = upload_dict["annotation_keys"]
    configs.MODEL_PATH = f"../results/models/{selected_model_name}"
    for index, cls_name in enumerate(configs.CLASS_NAMES):
        configs.PATCHCORE_OPTIONS[cls_name] = upload_dict["parchcore_options"][index]

    
    tester = Tester(configs)
    result = tester.run(
        dir_name = target_folder, 
        file_name = f"{selected_model_name}_{test_name}"
    )
    
    return return_template(request)
    
def return_template(request):
    update_upload_state()
    return templates.TemplateResponse("index.html",
    {
        "request": request,
        "train_image_upload_num":upload_dict["train_image_upload_num"],
        "valid_image_upload_num":upload_dict["valid_image_upload_num"],
        "test_image_upload_num":upload_dict["test_image_upload_num"],
        "train_annotation_upload":upload_dict["train_annotation_upload"],
        "valid_annotation_upload":upload_dict["valid_annotation_upload"],
        "annotation_keys":upload_dict["annotation_keys"],
        "parchcore_options":upload_dict["parchcore_options"],
        "model_list":upload_dict["model_list"],
        "test_list":upload_dict["test_list"],
    })