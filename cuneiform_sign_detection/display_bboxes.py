import sys
from pathlib import Path
from typing import Union

import matplotlib

from cuneiform_sign_detection.preprocessing.delete_corrupt_images_and_annotations import (
    delete_corrupt_images_and_annotations,
)
from cuneiform_sign_detection.preprocessing.validate_data import is_valid_file_size

matplotlib.use("tkAgg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


files_to_delete = []


def display_bboxes(
    first: Path,
    second: Union[Path, str] = "imgs",
    third: Union[Path, str] = "annotations",
) -> None:
    """
    Either two file paths or directory with subfolder names (optional)
    Args:
      first: data_directory or image_file_path
      second: subfolder_name of images_directory in data_directory or annotations_file_path
      third: subfolder_name of annotations_directory in data_directory

    """
    if first.is_file():
        assert isinstance(second, Path)
        assert second.is_file()
        show_img(first, second)
    elif first.is_dir():
        imgs_path = first / second
        annotations_path = first / third
        show_all_in_dir(imgs_path, annotations_path)
    else:
        raise ValueError(
            "arguments have to be either one directories with optional subfolder names or both single files"
        )


def show_all_in_dir(img_path: Path, annotations_path: Path) -> None:
    for image in img_path.iterdir():
        annotation = next(annotations_path.glob(f"gt_{image.stem}.txt"), None)
        if not annotation:
            raise FileNotFoundError(
                f"No annotations text file found corresponding to image: '{image.name}'"
            )
        show_img(image, annotation)


def on_press(image_path: Path):
    def on_press_(event):
        sys.stdout.flush()
        if event.key == "y":
            plt.close()
        elif event.key == "n":
            print(image_path.stem)
            files_to_delete.append(image_path.stem)
            plt.close()
        elif event.key == "escape":
            quit()

    return on_press_


def plot_bbox_with_img(image_path: Path, gt_path: Path) -> None:
    ground_truth_df = pd.read_csv(gt_path, header=None).infer_objects()
    image = Image.open(image_path)
    image = np.asarray(image)
    fig, ax = plt.subplots(figsize=(20, 20))
    fig.canvas.mpl_connect("key_press_event", on_press(image_path))
    ax.imshow(image)
    rectangles = dict()
    for counter, (index, row) in enumerate(ground_truth_df.iterrows()):
        bbox = row.array[:4]
        sign = row.array[-1] if len(row.array) == 5 else ""
        sign = f"{sign}-{counter}"
        for number in bbox:
            assert number >= 0
        rectangle = plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            fill=False,
            edgecolor="blue",
            alpha=0.8,
            linewidth=1,
        )
        rectangles[sign] = rectangle

    for r in rectangles:
        ax.add_artist(rectangles[r])
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width() / 2.0
        cy = ry + rectangles[r].get_height() / 2.0
        ax.annotate(
            r, (cx, cy), color="w", weight="bold", fontsize=6, ha="center", va="center"
        )
    plt.title(image_path)
    plt.show()


def show_img(image_path: Path, gt_path: Path) -> None:
    if not is_valid_file_size(image_path, False):
        print(f"Image '{image_path.stem}' has size 0 bytes. Please check")
    elif not is_valid_file_size(gt_path, False):
        print(
            f"Annotation '{gt_path.stem}' has size 0 bytes. Annotations and image will be deleted"
        )
        files_to_delete.append(image_path.stem)
    else:
        plot_bbox_with_img(image_path, gt_path)


if __name__ == "__main__":
    """
    display annotations boxes and images. Skip through images with pressing "ESC" or pressing "k" which will
    log image id to console. This way one can manually skip through training data and exclude data samples
    which are not good!
    """
    """
    #Display single image + annotation
    img = Path("../../temp/heidelberg+lmu+extracted/imgs/P336009.jpg")
    annotation = Path("../../temp/heidelberg+lmu/annotations/gt_P336009.txt")
    display_bboxes(img, annotation)
    quit()
    """
    data_path_str = "temp/lmu-extracted-2"
    data_path = Path(data_path_str)
    display_bboxes(data_path)
    # display_bboxes(data_path / Path("imgs/BM.33337.jpg"), data_path / Path("annotations/gt_BM.33337.txt"))
    delete_corrupt_images_and_annotations(data_path, files_to_delete)
