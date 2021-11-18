from pathlib import Path

import pytest

from cuneiform_sign_detection.preprocessing.delete_corrupt_images_and_annotations import (
    delete_corrupt_images_and_annotations,
)


@pytest.fixture
def setup_files(tmp_path):
    data = tmp_path / "data"
    annotations_directory = data / "annotations"
    imgs_directory = data / "imgs"

    [directory.mkdir() for directory in [data, annotations_directory, imgs_directory]]

    images = ["image_" + str(_id) for _id in range(3)]
    annotations = [f"gt_{image}.txt" for image in images] + ["gt_image_to_delete.txt"]

    for image in images:
        path = imgs_directory / image
        path.write_text("")

    for annotation in annotations:
        path = annotations_directory / annotation
        path.write_text("")

    return data, annotations_directory


@pytest.mark.parametrize("to_delete", [Path("to_delete.txt"), ["image_to_delete"]])
def test_delete_corrupt_images_and_annotations(setup_files, to_delete):
    data, annotations_directory = setup_files
    if isinstance(to_delete, Path):
        to_delete = data / to_delete
        to_delete.write_text("image_to_delete")

    assert len(list(annotations_directory.iterdir())) == 4
    assert next(annotations_directory.glob("*image_to_delete.txt"), None) is not None
    delete_corrupt_images_and_annotations(data, to_delete)
    assert len(list(annotations_directory.iterdir())) == 3
    assert next(annotations_directory.glob("*image_to_delete.txt"), None) is None
