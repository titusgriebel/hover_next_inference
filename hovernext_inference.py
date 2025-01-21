import os
import zarr
import imageio
import numpy as np
from glob import glob
from natsort import natsorted
import zipfile
import shutil
import subprocess

import json
CHECKPOINTS = [
    # "lizard_convnextv2_large",
    # "lizard_convnextv2_base",
    # "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    # "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]

DATASETS = [
    # "consep",
    # "cpm15",
    # "cpm17",
    # "cryonuseg",
    # "lizard",
    # "lynsec_he",
    # "lynsec_ihc",
    # "monusac",
    # "monuseg",
    # "nuclick",
    # "nuinsseg",
    "pannuke",
    # "puma",
    # "srsanet",
    # "tnbc",
]


def postprocess_inference(path):
    inst_path = os.path.join(path, "instance")
    os.makedirs(inst_path, exist_ok=True)
    sem_path = os.path.join(path, "semantic")
    os.makedirs(sem_path, exist_ok=True)
    for image_dir in natsorted(glob(os.path.join(path, "*"))):
        if not os.path.isdir(image_dir):
            continue
        
        #instance map
        array_path = os.path.join(image_dir, "pinst_pp")
        os.makedirs(array_path, exist_ok=True)
        with zipfile.ZipFile(os.path.join(image_dir, "pinst_pp.zip"), "r") as zip_file:
            zip_file.extractall(array_path)
        raw_pred = zarr.open(os.path.join(image_dir, "pinst_pp"), mode="r")
        instance_map = np.squeeze(raw_pred)
        
        imageio.imwrite(
            os.path.join(inst_path, f"{os.path.basename(image_dir)}.tiff"),
            instance_map,
            format="TIFF",
        )
        #class info
        json_path = os.path.join(image_dir, "class_inst.json")
        with open(json_path, 'r') as file:
            class_info = json.load(file)
        id_to_class = {int(k): v[0] for k, v in class_info.items()}
        semantic_map = np.zeros_like(instance_map)
        for instance_id, class_label in id_to_class.items():
            semantic_map[instance_map == instance_id] = class_label
        imageio.imwrite(
            os.path.join(sem_path, f"{os.path.basename(image_dir)}.tiff"),
            semantic_map,
            format="TIFF",
        )

        shutil.rmtree(image_dir)
        shutil.rmtree(array_path)


def run_inference(input_dir, output_dir):
    for dataset in ['pannuke']:
        for model in CHECKPOINTS:
            output_path = os.path.join(output_dir, "inference", dataset, model)
            input_path = os.path.join(input_dir, dataset, "loaded_testset", "eval_split", "test_images", "*")
            if os.path.exists(os.path.join(output_dir, 'results', dataset, model, 'ais_result.csv')):
                    print(f"Inference with HoVerNeXt model (type: {model}) on {dataset} dataset already done")
                    continue
            os.makedirs(output_path, exist_ok=True)
            args = [
                "--input",
                f"{input_path}",
                "--cp",
                f"{model}",
                "--output_root",
                f"{output_path}",
                "--tile_size",
                "512",
            ]
            command = [
                "python3",
                "/user/titus.griebel/u12649/hover_next_inference/main.py",
            ] + args
            print(
                f"Running inference with HoVerNeXt {model} model on {dataset} dataset..."
            )
            subprocess.run(command)
            print(
                f"Inference on {dataset} dataset with the HoVerNeXt model {model} successfully completed"
            )
            postprocess_inference(output_path)
            print("Predictions successfully converted to .tiff")


run_inference(
    input_dir="/mnt/lustre-grete/usr/u12649/data/semantic_data",
    output_dir="/mnt/lustre-grete/usr/u12649/models/hovernext_types",
)
