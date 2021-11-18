from pathlib import Path

import matplotlib

matplotlib.use("tkAgg")
from mmcv import Config
from mmocr.apis import init_detector, model_inference


if __name__ == "__main__":
    # from cuneiform_sign_detection.configs.fcenet_default import fcenet_cpu

    # configFile = fcenet_cpu.__file__
    cfg = Config.fromfile("logs/fcenet_cpu/2/fcenet_no_dcvn.py")

    checkpoint = "logs/fcenet_cpu/2/checkpoint.pth"

    to_predict_dir = "temp/to_predict"
    out_imgs = "temp/predictions"

    model = init_detector(cfg, checkpoint=checkpoint, device="cpu")
    if model.cfg.data.test["type"] == "ConcatDataset":
        model.cfg.data.test.pipeline = model.cfg.data.test["datasets"][0].pipeline

    for img_path in Path(to_predict_dir).iterdir():
        result = model_inference(model, str(img_path))
        model.show_result(
            img_path,
            result,
            out_file=f"{out_imgs}/prediction_{img_path.name}",
            show=True,
            thickness=2,
            bbox_color="red",
        )
        print(f"{img_path.name} completed")
