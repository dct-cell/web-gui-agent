"""Download datasets from HuggingFace."""
import argparse
import os
from pathlib import Path


def download_mind2web(data_dir: str):
    """Download Mind2Web dataset."""
    from datasets import load_dataset

    save_path = Path(data_dir) / "mind2web"
    if save_path.exists():
        print(f"Mind2Web already exists at {save_path}, skipping.")
        return

    print("Downloading Mind2Web training set...")
    ds = load_dataset("osunlp/Mind2Web", split="train")
    save_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_path / "train"))
    print(f"Saved {len(ds)} samples to {save_path / 'train'}")


def download_showui_web(data_dir: str):
    """Download ShowUI-web grounding dataset."""
    save_path = Path(data_dir) / "showui-web"
    if save_path.exists():
        print(f"ShowUI-web already exists at {save_path}, skipping.")
        return

    print("Downloading ShowUI-web dataset...")
    os.makedirs(save_path, exist_ok=True)
    os.system(
        f"huggingface-cli download showlab/ShowUI-web "
        f"--repo-type dataset --local-dir {save_path}"
    )
    print(f"Saved to {save_path}")


def download_screenspot(data_dir: str):
    """Download ScreenSpot evaluation dataset."""
    save_path = Path(data_dir) / "screenspot"
    if save_path.exists():
        print(f"ScreenSpot already exists at {save_path}, skipping.")
        return

    print("Downloading ScreenSpot dataset...")
    os.makedirs(save_path, exist_ok=True)
    os.system(
        f"huggingface-cli download KevinQHLin/ScreenSpot "
        f"--repo-type dataset --local-dir {save_path}"
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mind2web", "showui-web", "screenspot"],
        choices=["mind2web", "showui-web", "screenspot"],
    )
    args = parser.parse_args()

    download_fns = {
        "mind2web": download_mind2web,
        "showui-web": download_showui_web,
        "screenspot": download_screenspot,
    }
    for name in args.datasets:
        download_fns[name](args.data_dir)
