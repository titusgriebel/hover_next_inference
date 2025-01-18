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
    # "lizard_convnextv2_large",
    # "lizard_convnextv2_base",
    # "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    "pannuke_convnextv2_tiny_2",
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
    for image_dir in natsorted(glob(os.path.join(path, "*"))):
        if not os.path.isdir(image_dir):
            continue
        array_path = os.path.join(image_dir, "cls")
        os.makedirs(array_path, exist_ok=True)
        with zipfile.ZipFile(os.path.join(image_dir, f"{os.path.basename(image_dir)}_raw_512_cls.zip"), "r") as zip_file:
            zip_file.extractall(array_path)
        raw_pred = zarr.open(os.path.join(image_dir, "cls"), mode="r")
        pred = np.squeeze(raw_pred)
        if pred.size == 0:
            print(image_dir)
            continue
        semantic_mask = np.argmax(pred, axis=0)
        imageio.imwrite(
            os.path.join(path, f"{os.path.basename(image_dir)}.tiff"),
            semantic_mask,
            format="TIFF",
        )
        shutil.rmtree(image_dir)


def run_inference(input_dir, output_dir):
    for dataset in DATASETS:
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
                "--keep_raw",
            ]
            command = [
                "python3",
                "/user/titus.griebel/u12649/hover_next_inference/main.py",
            ] + args
            print(
                f"Running inference with HoVerNeXt {model} model on {dataset} dataset..."
            )
            # subprocess.run(command)
            print(
                f"Inference on {dataset} dataset with the HoVerNeXt model {model} successfully completed"
            )
            postprocess_inference(output_path)
            print("Predictions successfully converted to .tiff")


run_inference(
    input_dir="/mnt/lustre-grete/usr/u12649/data/final_test",
    output_dir="/mnt/lustre-grete/usr/u12649/models/hovernext_types",
)
