import copy

from mmocr.datasets import build_dataset


def create_val_dataset(cfg):
    val_dataset = copy.deepcopy(cfg.data.val)
    if cfg.data.train["type"] == "ConcatDataset":
        train_pipeline = cfg.data.train["datasets"][0].pipeline
    else:
        train_pipeline = cfg.data.train.pipeline

    if val_dataset["type"] == "ConcatDataset":
        for dataset in val_dataset["datasets"]:
            dataset.pipeline = train_pipeline
    else:
        val_dataset.pipeline = train_pipeline
    return build_dataset(val_dataset)
