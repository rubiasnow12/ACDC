import os
from pathlib import Path
import pickle
import yaml
from typing import List, Dict
from pprint import pprint

import numpy as np
from torch.utils.data import DataLoader

import monai
from monai.data import CacheDataset, Dataset, list_data_collate, pad_list_data_collate
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd, RandSpatialCropd,
    CenterSpatialCropd, Lambdad, ToTensord, Compose, ResizeWithPadOrCropd
)

# 增加：模块级 z-score 函数（可被 multiprocessing pickle）
def zscore(x):
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

# 增加：模块级时间维处理函数（可被 pickle）
def collapse_image_mean(x):
    """若 image 含多帧（first dim 为 time/channel），对时间维取均值，确保返回 shape (1, H, W, ...)"""
    if x is None:
        return None
    arr = np.asarray(x)
    # 当存在 time/channel 维且长度>1 时，对第0维求均值；若是 3D 则补 channel 维
    if arr.ndim >= 4:
        arr = np.mean(arr, axis=0, keepdims=True)
    elif arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32)

def select_label_center(x):
    """若 label 有多帧，则取中心帧并返回 shape (1, H, W, ...)；兼容 None"""
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim >= 4:
        idx = arr.shape[0] // 2
        sel = arr[idx]
        arr = np.expand_dims(sel, 0)
    elif arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32)

def find_image_label_pairs(patient_dir: Path) -> List[Dict]:
    """
    查找并返回该 patient 下的 image/label 对列表（每个 item: {"image": path, "label": path, "patient_id": id}）
    说明：不同 ACDC 数据组织方式不同。这里给出通用策略：
      - 优先寻找 *_gt* / *seg* / *_mask* 的文件作为 label；
      - 其余 .nii/.nii.gz 文件视为 image（排除 Info.cfg）。
    如果您的数据组织为 patient/{image.nii.gz, label.nii.gz}，此函数能工作；
    若组织为 time/frame 则需要修改筛选逻辑（按文件名后缀或子目录配对）。
    """
    imgs = []
    labels = []
    for f in sorted(patient_dir.glob("*")):
        if f.name.lower().endswith((".nii", ".nii.gz", ".mha", ".mhd")) and f.is_file():
            name = f.name.lower()
            if ("seg" in name) or ("gt" in name) or ("mask" in name):
                labels.append(f.as_posix())
            elif f.name != "Info.cfg":
                imgs.append(f.as_posix())
    pairs = []
    # 简单配对策略：如果 labels数量==imgs数量一一对应，否则按 idx 配对第一个 label 到所有 imgs（常见：整体 label）
    if len(labels) == len(imgs) and len(imgs) > 0:
        for im, lb in zip(imgs, labels):
            pairs.append({"image": im, "label": lb, "patient_id": patient_dir.name})
    elif len(labels) == 1 and len(imgs) >= 1:
        for im in imgs:
            pairs.append({"image": im, "label": labels[0], "patient_id": patient_dir.name})
    elif len(imgs) == 1 and len(labels) == 0:
        # 没有 label 情形（例如测试集无标注），label 设为 None
        pairs.append({"image": imgs[0], "label": None, "patient_id": patient_dir.name})
    else:
        # 回退：把首个 nii 当作 image，首个 seg 当作 label（如果存在）
        if imgs:
            pairs.append({"image": imgs[0], "label": labels[0] if labels else None, "patient_id": patient_dir.name})
    return pairs

def build_transforms(cfg: dict, is_training: bool):
    pp = cfg["preprocess"]
    spatial_ndim = int(pp.get("spatial_ndim", 3))
    crop_method = pp.get("crop_method", "center")
    crop_size_2d = tuple(pp.get("crop_size_2d", [256, 256]))
    crop_size_3d = tuple(pp.get("crop_size_3d", [128, 128, 128]))
    target_spacing = pp.get("target_spacing", None)
    roi_size = crop_size_2d if spatial_ndim == 2 else crop_size_3d
    keys = ["image", "label"]

    tr = [
        LoadImaged(keys=keys, reader="ITKReader"),
        EnsureChannelFirstd(keys=keys),
        # 先把多帧 image/label 归一为固定 channel 形式（image: mean collapse, label: center frame）
        Lambdad(keys=["image"], func=collapse_image_mean),
        Lambdad(keys=["label"], func=select_label_center),
    ]
    if target_spacing:
        tr.append(Spacingd(keys=keys, pixdim=tuple(target_spacing), mode=("bilinear", "nearest")))
    tr.append(CropForegroundd(keys=keys, source_key="image"))
    # 先标准化强度
    tr.append(Lambdad(keys=["image"], func=zscore))

    # 先做随机/中心裁切（尽量保留 ROI），然后确保最终尺寸一致：ResizeWithPadOrCropd
    def spatial_crop_op():
        if is_training and crop_method == "random":
            return RandSpatialCropd(keys=keys, roi_size=roi_size, random_center=True, random_size=False)
        else:
            return CenterSpatialCropd(keys=keys, roi_size=roi_size)

    tr.append(spatial_crop_op())
    tr.append(ResizeWithPadOrCropd(keys=keys, spatial_size=roi_size))
    tr.append(ToTensord(keys=keys))
    return Compose(tr)

def _load_yaml_flexible(path: str):
    """尝试以 utf-8 打开 yaml，若失败回退到 gbk（Windows 下常见）"""
    path = Path(path)
    try:
        with path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except UnicodeDecodeError:
        with path.open('r', encoding='gbk', errors='ignore') as f:
            return yaml.safe_load(f)

def make_dataloaders(config_path: str = "configs/config.yaml"):
    # with open(config_path, "r") as f:
    #     cfg = yaml.safe_load(f)
    cfg = _load_yaml_flexible(config_path)

    root = Path(cfg["data"]["root"])
    splits_file = Path(cfg["data"]["splits"])
    with open(splits_file, "rb") as f:
        splits = pickle.load(f)

    # 构建 sample 列表（按 patient 划分）
    def build_samples(patient_list):
        samples = []
        for pid in patient_list:
            pdir = root / pid
            if not pdir.exists():
                print(f"Warning: {pdir} not exist, skipping")
                continue
            pairs = find_image_label_pairs(pdir)
            samples.extend(pairs)
        return samples

    train_samples = build_samples(splits["train"])
    val_samples = build_samples(splits["val"])
    test_samples = build_samples(splits["test"])

    print("Counts:", len(train_samples), len(val_samples), len(test_samples))

    train_trans = build_transforms(cfg, is_training=True)
    val_trans = build_transforms(cfg, is_training=False)

    # 使用 CacheDataset 加速（若内存有限可换 Dataset）
    train_ds = CacheDataset(data=train_samples, transform=train_trans, cache_rate=1.0, num_workers=cfg["dataloader"]["num_workers"])
    val_ds = CacheDataset(data=val_samples, transform=val_trans, cache_rate=1.0, num_workers=cfg["dataloader"]["num_workers"])
    test_ds = CacheDataset(data=test_samples, transform=val_trans, cache_rate=0.0, num_workers=cfg["dataloader"]["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        collate_fn=pad_list_data_collate,  # 改成 pad_list_data_collate 更健壮（和 ResizeWithPadOrCropd 双保险）
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["dataloader"]["val_batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        collate_fn=pad_list_data_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["dataloader"]["val_batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        collate_fn=pad_list_data_collate,
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    tl, vl, te = make_dataloaders("configs/config.yaml")
    print("Example batch from train:")
    # 为避免 Windows 多进程 pickling 问题，演示时使用单进程 DataLoader 从同一 dataset 读取一个 batch
    from torch.utils.data import DataLoader as _DL
    single_loader = _DL(tl.dataset, batch_size=tl.batch_size if hasattr(tl, "batch_size") else 1, shuffle=True, num_workers=0, collate_fn=list_data_collate)
    for b in single_loader:
        print("batch keys:", b.keys() if isinstance(b, dict) else (b[0].keys() if isinstance(b, list) else "unknown"))
        print("image shape:", b["image"].shape if "image" in b else "n/a")
        break