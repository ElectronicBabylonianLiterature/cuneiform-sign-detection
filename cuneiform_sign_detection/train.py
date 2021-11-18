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
    # datasets.append(create_val_dataset(cfg))

    return datasets


def train(cfg, datasets) -> None:
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=False, validate=True)


def log_data(from_path: Path, to_file: Path) -> int:
    samples = [file.name for file in from_path.iterdir()]
    to_file.write_text("\n".join(samples))
    return len(samples)


if __name__ == "__main__":
    """
    data format is coco and in directory ../data

    Disclaimer batch/sample size for validation test set have to be one for hmean eval
    to work !
    """

    from cuneiform_sign_detection.configs.fcenet_default import (
        fcenet_no_dcvn as model_config,
    )

    configFile = model_config.__file__
    load_from_checkpoint = "./checkpoints/fcenet_cpu.pth"

    data_path = "./data"

    cfg = configure_cfg(configFile, data_path, load_from_checkpoint)

    training_data = Path(data_path) / "imgs" / "training"

    training_data_samples = log_data(
        training_data, (Path(cfg.work_dir) / "training_data.txt")
    )
    validation_path = Path(data_path) / "imgs" / "validation"
    validation_data_samples = 0
    if validation_path.exists() and any(validation_path.iterdir()):
        validation_data_samples = log_data(
            validation_path, (Path(cfg.work_dir) / "validation_data.txt")
        )

    summary = f"Training Data Points: {training_data_samples}\nValidation Data Samples: {validation_data_samples}"

    (Path(cfg.work_dir) / "summary.txt").write_text(summary)
    cfg.dump(f"{cfg.work_dir}/{Path(configFile).name}")

    print(f"Config:\n{cfg.pretty_text}")

    datasets = prepare_dataset(cfg)
    train(cfg, datasets)
