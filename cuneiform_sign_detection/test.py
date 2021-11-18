import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector


def bbox_from_polygon(box):
    if len(box) > 8:
        y = box[::2]
        x = box[1::2]
        y1, x1 = min(y), min(x)
        y2, x2 = y1, max(x)
        y3, x3 = max(y), x2
        y4, x4 = y3, x1
        box = [y1, x1, y2, x2, y3, x3, y4, x4]
        return box
    else:
        ValueError("Looks like its already a bbox")


def test(
    cfg,
    checkpoint,
    out=None,
    show=True,
    show_dir="../temp/test/imgs",
    show_score_thr=0.3,
    eval="bbox",
):
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    test_loader_cfg = {
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get("test_dataloader", {}),
    }
    cfg.model.train_cfg = None
    data_loader = build_dataloader(dataset, **test_loader_cfg, workers_per_gpu=2)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, show, show_dir, show_score_thr)

    output_bboxes = []
    for output in outputs:
        result = []
        for bbox in output["boundary_result"]:
            result.append([*bbox_from_polygon(bbox[:-1]), bbox[-1]])
        output_bboxes.append({"boundary_result": result})
    outputs = output_bboxes
    if out:
        print(f"\nwriting results to {out}")
        mmcv.dump(outputs, out)
    if eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        # hard-code way to remove EvalHook args
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
        print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    from cuneiform_sign_detection.configs.fcenet_default import fcenet_no_dcvn as model

    configFile = model.__file__
    checkpoint = "../logs/fcenet_dcvn/2/epoch_300.pth"
    cfg = Config.fromfile(configFile)
    cfg.data.test_dataloader = dict(samples_per_gpu=1)
    cfg.data.test.ann_file = "../data/instances_validation.json"
    cfg.evaluation = dict(interval=1, metric=["hmean-iou", "hmean-ic13"])

    test(cfg, checkpoint)
