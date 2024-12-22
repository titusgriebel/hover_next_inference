import os
import zarr
import imageio
import numpy as np
from glob import glob
from natsort import natsorted
import zipfile
import shutil
import subprocess

CHECKPOINTS = [
    "lizard_convnextv2_large",
    "lizard_convnextv2_base",
    "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]
DATASETS = [
    "cpm15",
    "cpm17",
    "cryonuseg",
    "janowczyk",
    "lizard",
    "lynsec",
    "monusac",
    "monuseg",
    "nuinsseg",
    "pannuke",
    "puma",
    "tnbc",
]


def postprocess_inference(path):
    for image_dir in natsorted(glob(os.path.join(path, "*"))):
        if not os.path.isdir(image_dir):
            continue
        array_path = os.path.join(image_dir, "pinst_pp")
        os.makedirs(array_path, exist_ok=True)
        with zipfile.ZipFile(os.path.join(image_dir, "pinst_pp.zip"), "r") as zip_file:
            zip_file.extractall(array_path)
        raw_pred = zarr.open(os.path.join(image_dir, "pinst_pp"), mode="r")
        pred = np.squeeze(raw_pred)
        imageio.imwrite(
            os.path.join(path, f"{os.path.basename(image_dir)}.tiff"),
            pred,
            format="TIFF",
        )
        shutil.rmtree(image_dir)


def run_inference(input_dir, output_dir):
    for dataset in DATASETS:
        for model in CHECKPOINTS:
            output_path = os.path.join(output_dir, dataset, model)
            input_path = os.path.join(input_dir, dataset, "loaded_testset/images/*")
            if os.path.exists(output_path):
                if len(os.listdir(output_path)) > 0:
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
            print("Predictions succesfully converted to .tiff")


run_inference(
    input_dir="/mnt/lustre-grete/usr/u12649/scratch/data/final_test",
    output_dir="/mnt/lustre-grete/usr/u12649/scratch/models/hovernext/inference",
)
