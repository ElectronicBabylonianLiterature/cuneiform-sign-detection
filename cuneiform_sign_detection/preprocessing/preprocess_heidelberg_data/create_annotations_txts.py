import shutil
from ast import literal_eval
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from PIL import Image

from cuneiform_sign_detection.bounding_boxes import BoundingBoxes
from cuneiform_sign_detection.preprocessing.delete_corrupt_images_and_annotations import (
    delete_corrupt_images_and_annotations,
)
from cuneiform_sign_detection.preprocessing.validate_data import is_valid_data

from cuneiform_sign_detection.utils.path import (
    create_directory,
)


def process(
    save_data_path: Path, annotations_df: pd.DataFrame, vat_image_directory: Path
) -> None:
    """
    bbox format is [xmin, ymin, xmax, ymax]
    """
    image_path = save_data_path / "imgs"
    annotations_path = save_data_path / "annotations"
    create_directory(save_data_path, overwrite=True)
    create_directory(image_path)
    create_directory(annotations_path)

    annotations_df.bbox = annotations_df.bbox.apply(literal_eval).apply(np.array)

    failed_downloads = []
    cdlis = set(annotations_df.tablet_CDLI)
    print("Number of images: ", len(cdlis), "\n")
    for cdli in cdlis:
        cdli_annotations = annotations_df[annotations_df.tablet_CDLI == cdli]
        bboxes = cdli_annotations["bbox"].to_numpy()
        bounding_boxes = BoundingBoxes.from_two_vertices(cdli, bboxes)
        bounding_boxes.create_ground_truth_txt(annotations_path)

        download_path = f"https://cdli.ucla.edu/dl/photo/{cdli}.jpg"
        if "VAT" not in cdli:  # VAT images downloaded not download from cdli
            try:
                im = Image.open(urlopen(download_path))
                im.save(f"{image_path}/{cdli}.jpg")
            except HTTPError:
                failed_downloads.append(cdli)
                print(f"Failed: {cdli}")
                continue
        print(f"Success: {cdli}")

    for vat_image in vat_image_directory.iterdir():
        shutil.copy(vat_image, image_path / vat_image.name)

    print("---------------Failed Downloads-------------------------")
    print("\n".join(failed_downloads))
    delete_corrupt_images_and_annotations(save_data_path, failed_downloads)
    is_valid_data(save_data_path)


if __name__ == "__main__":
    """
    Images fetched from cdli website: https://cdli.ucla.edu/dl/photo/{cdli_number}.jpg"
    VAT images have to be downloaded manually from: https://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/model_weights/VAT_train_images.zip
    Link from https://github.com/CompVis/cuneiform-sign-detection-dataset
    """
    DOWNLOAD_DIRECTORY = "heidelberg"
    VAT_DIRECTORY = "datasets/VAT_images"
    save_data_path = Path("temp") / DOWNLOAD_DIRECTORY
    train_bbox_annotations = pd.read_csv(
        "cuneiform_sign_detection/preprocessing/preprocess_heidelberg_data/annotations_csv/bbox_annotations_train_full.csv"
    )
    full_df = train_bbox_annotations.append(
        pd.read_csv(
            "cuneiform_sign_detection/preprocessing/preprocess_heidelberg_data/annotations_csv/bbox_annotations_test_full.csv"
        )
    )
    process(save_data_path, full_df, vat_image_directory=Path(VAT_DIRECTORY))
