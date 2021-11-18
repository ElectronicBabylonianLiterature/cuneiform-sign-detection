import os
import random
import shutil
from pathlib import Path
from typing import Tuple, Sequence

from cuneiform_sign_detection.preprocessing.create_coco_json import create_coco_json
from cuneiform_sign_detection.preprocessing.validate_data import is_valid_data


def copy_image_and_corresponding_annotation(
    images_path: Sequence[Path],
    output_folder_annotations: Path,
    output_folder_imgs: Path,
    annotations_path_copy_from: Path,
):
    for file in images_path:
        shutil.copyfile(file, output_folder_imgs / file.name)
        annotation_path = next(annotations_path_copy_from.glob(f"gt_{file.stem}.txt"))
        shutil.copyfile(
            annotation_path, output_folder_annotations / annotation_path.name
        )


def split_data(
    size: int,
    size_test: float,
    size_val: float,
    create_test: bool,
    data: Sequence[Path],
) -> Tuple[Sequence[Path], ...]:
    size_test = int(size_test * size)
    size_val = int(size_val * size)

    if create_test:
        test = data[:size_test]
        validation = data[size_test : size_test + size_val]
        training = data[size_test + size_val :]
    else:
        validation = data[:size_val]
        training = data[size_val:]
        test = []
    return training, validation, test


def prepare_data_coco_format(
    data_path: Path,
    output_path: Path,
    create_test=True,
    split=0.15,
    random_seed: int = 1,
) -> None:
    random.seed(random_seed)
    images_path = data_path / "imgs"
    annotations_path = data_path / "annotations"
    is_valid_data(data_path)
    data = sorted(images_path.iterdir(), key=lambda file: file.name)
    random.shuffle(data)
    size = len(data)

    training, validation, test = split_data(size, split, split, create_test, data)

    print(size)
    print("Training :", len(training))
    print("Validation :", len(validation))
    print("Test: ", len(test))

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    annotations_training_path = output_path / "annotations/training"
    annotations_validation_path = output_path / "annotations/validation"
    imgs_training_path = output_path / "imgs/training"
    imgs_validation_path = output_path / "imgs/validation"

    for path in [
        annotations_training_path,
        annotations_validation_path,
        imgs_training_path,
        imgs_validation_path,
    ]:
        path.mkdir(parents=True)

    copy_image_and_corresponding_annotation(
        training, annotations_training_path, imgs_training_path, annotations_path
    )
    copy_image_and_corresponding_annotation(
        validation, annotations_validation_path, imgs_validation_path, annotations_path
    )

    if test:
        annotations_test_path = output_path / "annotations/test"
        imgs_test_path = output_path / "imgs/test"
        annotations_test_path.mkdir(parents=True)
        imgs_test_path.mkdir(parents=True)

        copy_image_and_corresponding_annotation(
            test, annotations_test_path, imgs_test_path, annotations_path
        )
        create_coco_json(output_path, ["training", "validation", "test"], output_path)

    create_coco_json(output_path, ["training"], output_path)


if __name__ == "__main__":
    data_path = Path("temp/total")
    output_path = Path("data")

    prepare_data_coco_format(data_path, output_path, False, 0.0)
