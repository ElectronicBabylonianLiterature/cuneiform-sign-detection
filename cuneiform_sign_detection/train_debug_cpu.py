import os.path as osp
from pathlib import Path

import mmcv
from mmcv import Config
from mmocr.apis import train_detector
from mmocr.datasets import build_dataset
from mmocr.models import build_detector

from cuneiform_sign_detection.utils.calculate_mean_and_std import (
    mean_and_std_from_data_path,
)
from cuneiform_sign_detection.utils.create_val_dataset import create_val_dataset
from cuneiform_sign_detection.utils.display_images_after_pipeline import (
    display_images_after_pipeline,
)
from cuneiform_sign_detection.utils.path import (
    log_version_increment,
)


def configure_cfg(
    config_file_path: str,
    data_path: str,
    load_from_checkpoint=None,
    resume_from_checkpoint=None,
    gpu_ids=range(1),
    log_directory="./logs",
):

    cfg = Config.fromfile(config_file_path)
    mean, std = mean_and_std_from_data_path(data_path)
    print("------------Mean and Std------------------")
    print(f"Mean: {str(mean)}")
    print(f"Std: {str(std)}")
    print("-------------------------------------------")

    cfg.img_norm_cfg = dict(mean=mean, std=std, to_rgb=True)

    log_dir = Path(log_directory) / Path(config_file_path).stem
    model_version = log_version_increment(log_dir)

    cfg.work_dir = str(log_dir / str(model_version))
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if load_from_checkpoint and resume_from_checkpoint:
        raise ValueError("Either specifcy checkpoint or resume_from parameter not both")

    cfg.load_from = load_from_checkpoint
    cfg.resume_from = resume_from_checkpoint
    cfg.gpu_ids = gpu_ids
    return cfg


def prepare_dataset(cfg):
    datasets = [build_dataset(cfg.data.train)]
    datasets.append(create_val_dataset(cfg))

    return datasets


def train(cfg, datasets) -> None:
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == "__main__":
    """
    User input after starting script describing experiment will be used for logging
    Press double enter after comment to start training
    data format is coco and in directory ../data
    """

    from cuneiform_sign_detection.configs.fcenet_default import fcenet_dcvn

    configFile = fcenet_dcvn.__file__
    load_from_checkpoint = "./checkpoints/fcenet_dcvn.pth"
    data_path = "./data"

    comment = ""  # user_input()
    cfg = configure_cfg(configFile, data_path, load_from_checkpoint)

    print(f"Config:\n{cfg.pretty_text}")
    (Path(cfg.work_dir) / "summary.txt").write_text(comment)
    cfg.dump(f"{cfg.work_dir}/{Path(configFile).name}")

    datasets = prepare_dataset(cfg)
    display_images_after_pipeline(datasets[0])
    # train(cfg, datasets)
