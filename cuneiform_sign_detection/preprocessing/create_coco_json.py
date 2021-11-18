# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from functools import partial

import mmcv
from mmocr.utils import convert_annotations, list_from_file


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.
    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory
    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]
    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, "*" + suffix)))

    files = []
    for img_file in imgs_list:
        gt_file = gt_dir + "/gt_" + osp.splitext(osp.basename(img_file))[0] + ".txt"
        files.append((img_file, gt_file))
    assert len(files), f"No images found in {img_dir}"
    print(f"Loaded {len(files)} images from {img_dir}")

    return files


def collect_annotations(files, dataset, nproc=1):
    """Collect the annotation information.
    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        dataset(str): The dataset name, icdar2015 or icdar2017
        nproc(int): The number of process to collect annotations
    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(dataset, str)
    assert dataset
    assert isinstance(nproc, int)

    load_img_info_with_dataset = partial(load_img_info, dataset=dataset)
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info_with_dataset, files, nproc=nproc
        )
    else:
        images = mmcv.track_progress(load_img_info_with_dataset, files)

    return images


def load_img_info(files, dataset):
    """Load the information of one image.
    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)
        dataset(str): Dataset name, icdar2015 or icdar2017
    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)
    assert isinstance(dataset, str)
    assert dataset

    img_file, gt_file = files
    # read imgs with ignoring orientations
    img = mmcv.imread(img_file, "unchanged")

    if dataset == "icdar2017":
        gt_list = list_from_file(gt_file)
    elif dataset == "icdar2015":
        gt_list = list_from_file(gt_file, encoding="utf-8-sig")
    else:
        raise NotImplementedError(f"Not support {dataset}")

    anno_info = []
    for line in gt_list:
        # each line has one ploygen (4 vetices), and others.
        # e.g., 695,885,866,888,867,1146,696,1143,Latin,9
        line = line.strip()
        strs = line.split(",")

        category_id = 1
        bbox = [int(x) for x in strs[0:4]]

        iscrowd = 0

        x1, y1 = bbox[0], bbox[1]
        x2, y2 = x1 + bbox[2], y1
        x3, y3 = x2, y2 + bbox[3]
        x4, y4 = x1, y3
        segmentation = [x1, y1, x2, y2, x3, y3, x4, y4]

        area = bbox[2] * bbox[3]
        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            segmentation=[segmentation],
        )
        anno_info.append(anno)
    split_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(split_name, osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(split_name, osp.basename(gt_file)),
    )
    return img_info


def create_coco_json(icdar_path, split_list, out_dir, dataset="icdar2015", nproc=1):
    img_dir = osp.join(icdar_path, "imgs")
    gt_dir = osp.join(icdar_path, "annotations")

    set_name = {}
    for split in split_list:
        set_name.update({split: "instances_" + split + ".json"})
        assert osp.exists(osp.join(img_dir, split))

    for split, json_name in set_name.items():
        print(f"Converting {split} into {json_name}")
        with mmcv.Timer(print_tmpl="It takes {}s to convert icdar annotation"):
            files = collect_files(osp.join(img_dir, split), osp.join(gt_dir, split))
            image_infos = collect_annotations(files, dataset, nproc=nproc)
            convert_annotations(image_infos, osp.join(out_dir, json_name))
