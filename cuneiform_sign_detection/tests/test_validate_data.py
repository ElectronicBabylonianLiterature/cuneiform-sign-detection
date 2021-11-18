from pathlib import Path

import pytest

from cuneiform_sign_detection.preprocessing.validate_data import is_valid_data


@pytest.fixture
def setup_files(tmp_path):
    data = tmp_path / "data"
    annotations_directory = data / "annotations"
    imgs_directory = data / "imgs"

    [directory.mkdir() for directory in [data, annotations_directory, imgs_directory]]

    images = ["image_" + str(_id) for _id in range(3)]
    annotations = [f"gt_{image}.txt" for image in images]

    for image in images:
        path = imgs_directory / image
        path.write_text("test-image")

    for annotation in annotations:
        path = annotations_directory / annotation
        path.write_text("test-annotation")

    return data


@pytest.mark.parametrize(
    "invalid_file", [Path("annotations/gt_invalid.txt"), Path("imgs/image_invalid.txt")]
)
def test_invalid_validate_data_(setup_files, invalid_file):
    data = setup_files
    invalid_file = data / invalid_file
    invalid_file.write_text("invalid")
    with pytest.raises(FileNotFoundError):
        is_valid_data(data)


def test_valid_validate_data(setup_files):
    data = setup_files
    assert is_valid_data(data) is True
