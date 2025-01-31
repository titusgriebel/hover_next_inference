import os
import zarr
import imageio
import numpy as np
import zipfile
import shutil
import subprocess
import argparse
import json
from glob import glob
from natsort import natsorted

CHECKPOINTS = [
    "lizard_convnextv2_large",
    "lizard_convnextv2_base",
    "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]


def postprocess_inference(path, semantic):
    for image_dir in natsorted(glob(os.path.join(path, "*"))):
        if not os.path.isdir(image_dir):
            continue
        array_path = os.path.join(image_dir, "pinst_pp")
        os.makedirs(array_path, exist_ok=True)
        if not os.path.exists(os.path.join(image_dir, "pinst_pp.zip")):
            shutil.rmtree(image_dir)
        with zipfile.ZipFile(os.path.join(image_dir, "pinst_pp.zip"), "r") as zip_file:
            zip_file.extractall(array_path)
        raw_pred = zarr.open(os.path.join(image_dir, "pinst_pp"), mode="r")
        instance_map = np.squeeze(raw_pred)
        if semantic:
            json_path = os.path.join(image_dir, "class_inst.json")
            with open(json_path, 'r') as file:
                class_info = json.load(file)
            id_to_class = {int(k): v[0] for k, v in class_info.items()}
            semantic_map = np.zeros_like(instance_map)
            for instance_id, class_label in id_to_class.items():
                semantic_map[instance_map == instance_id] = class_label
            imageio.imwrite(
                os.path.join(path, f"{os.path.basename(image_dir)}.tiff"),
                semantic_map,
                format="TIFF",
            )
            if os.path.exists(image_dir):
                shutil.rmtree(image_dir)
            if os.path.exists(array_path):
                shutil.rmtree(array_path)
        else:
            imageio.imwrite(
                os.path.join(path, f"{os.path.basename(image_dir)}.tiff"),
                instance_map,
                format="TIFF",
            )
            if os.path.exists(image_dir):
                shutil.rmtree(image_dir)
            if os.path.exists(array_path):
                shutil.rmtree(array_path)
            continue


def run_inference(input_dir, output_dir, dataset, semantic, checkpoint):
    output_path = os.path.join(output_dir, "inference", dataset, checkpoint)
    input_path = os.path.join(input_dir, dataset, "eval_split", "test_images", "*")
    if os.path.exists(os.path.join(output_dir, 'results', dataset, checkpoint, 
                                   f'{dataset}_hovernext_{checkpoint}_ais_result.csv')):
        print(f"Inference with HoVerNeXt model (type: {checkpoint}) on {dataset} dataset already done")
        return
    os.makedirs(output_path, exist_ok=True)
    if dataset in ["pannuke", "srsanet", "nuclick"]:
        tile_size = 256
    else:
        tile_size = 512
    args = [
        "--input",
        f"{input_path}",
        "--cp",
        f"{checkpoint}",
        "--output_root",
        f"{output_path}",
        "--pp_tiling", "10",
        "--tile_size",
        f"{tile_size}",
    ]
    command = [
        "python3",
        "/user/titus.griebel/u12649/hover_next_inference/main.py",
    ] + args
    print(
        f"Running inference with HoVerNeXt {checkpoint} model on {dataset} dataset..."
    )
    subprocess.run(command)
    print(
        f"Inference on {dataset} dataset with the HoVerNeXt model {checkpoint} successfully completed"
    )
    postprocess_inference(output_path, semantic=semantic)
    print("Predictions successfully converted to .tiff")


def get_hnxt_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="The path to the datasets")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="The path to the datasets")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="The path to the datasets")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="The checkpoint to use for inference")
    parser.add_argument("--semantic", action="store_true", help="The path to the datasets")
    args = parser.parse_args()
    return args


def main():
    args = get_hnxt_args()
    run_inference(
        input_dir=args.input,
        output_dir=args.output_path,
        datasets=args.dataset,
        semantic=args.semantic,
        checkpoints=args.checkpoint,
    )


if __name__ == "__main__":
    main()
