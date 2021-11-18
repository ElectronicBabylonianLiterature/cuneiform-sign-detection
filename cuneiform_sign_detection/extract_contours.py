# pyre-ignore-all-errors[16]
import shutil
from pathlib import Path
from typing import List, Tuple, Sequence, Union

import attr

import cv2
import numpy as np
import pandas as pd

from cuneiform_sign_detection.bounding_boxes import BoundingBoxes
from cuneiform_sign_detection.utils.path import create_directory


@attr.s(auto_attribs=True, frozen=True)
class Point:
    x: int
    y: int


@attr.s(auto_attribs=True, frozen=True)
class Rectangle:
    top_left: Point
    bottom_right: Point

    def to_list(self):
        return [
            self.top_left.x,
            self.top_left.y,
            self.bottom_right.x,
            self.bottom_right.y,
        ]


def check_if_bbox_in_countour(img: np.ndarray, bboxes: np.ndarray) -> bool:
    intersection = np.logical_and(img, bboxes)
    return np.any(intersection)  # pyre-ignore[7]


def bbox_to_vec(bboxes: np.ndarray) -> List[Tuple[int, int]]:
    xy = []
    for box in bboxes:
        for i in range(0, len(box))[::2]:
            xy.append((int(box[i]), int(box[i + 1])))
    return xy


def crop_image(
    img: np.ndarray, contours: np.ndarray, index: int, bboxes: np.ndarray
) -> Union[Tuple[np.ndarray, Rectangle], Tuple]:
    cimg = np.zeros_like(img)
    bboxes_img = np.zeros_like(img)
    cv2.drawContours(cimg, contours, index, 1, -1)
    # cv2.imshow('1', cv2.resize(cimg*255, (960, 540)))
    # cv2.waitKey(0)
    bboxes_vec = bbox_to_vec(bboxes)
    for x, y in bboxes_vec:
        bboxes_img[y, x] = 1
    # cv2.imshow('2', cv2.resize(bboxes_img*255, (960, 540)))
    # cv2.waitKey(0)
    if check_if_bbox_in_countour(cimg, bboxes_img):
        pts = np.where(cimg == 1)
        padding = 15

        top_y = min(pts[0]) - padding
        bottom_y = max(pts[0]) + padding
        bottom_x = min(pts[1]) - padding
        top_x = max(pts[1]) + padding
        top_y = max(top_y, 0)
        bottom_x = max(bottom_x, 0)
        top_x = min(top_x, img.shape[1])
        bottom_y = min(bottom_y, img.shape[0])

        crop_img = img[top_y:bottom_y, bottom_x:top_x]
        new_coordinates = Rectangle(Point(bottom_x, top_y), Point(top_x, bottom_y))
        return crop_img, new_coordinates
    else:
        return tuple()


def extract_obverse_reverse(
    img: np.ndarray, bboxes: np.ndarray
) -> List[Tuple[np.ndarray, Rectangle]]:
    img_area = img.shape[0] * img.shape[1]
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, bw_img = cv2.threshold(im_gray, 200, 255, cv2.THRESH_OTSU)
    # bw_img = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_index = {}

    for i, contour in enumerate(contours):
        contour_index[i] = cv2.contourArea(contour)

    sorted_index = sorted(contour_index, key=contour_index.get, reverse=True)
    largest_contour = contour_index[sorted_index[0]]

    results = []
    if largest_contour >= img_area * 0.025:
        for i in sorted_index[0:20]:
            if contour_index[i] >= 0.2 * largest_contour:
                result = crop_image(img, contours, i, bboxes)
                if result:
                    results.append(result)
    return results


def annotation_to_numpy(path: Path) -> np.ndarray:
    annotation = pd.read_csv(path, header=None)
    return np.array([x[:4] for x in annotation.to_numpy()])


def is_rec1_in_rec2(rec1: Rectangle, rec2: Rectangle) -> bool:
    result = []
    result.append(True) if rec1.top_left.x >= rec2.top_left.x else result.append(False)
    result.append(True) if rec1.top_left.y >= rec2.top_left.y else result.append(False)
    result.append(
        True
    ) if rec1.bottom_right.x <= rec2.bottom_right.x else result.append(False)
    result.append(
        True
    ) if rec1.bottom_right.y <= rec2.bottom_right.y else result.append(False)

    return True if all(result) else False


def get_bboxes_belonging_to_crop(
    bboxes: np.ndarray, cropped_img_coordinates: Rectangle
):
    result = []
    for row in range(0, bboxes.shape[0]):
        bbox = bboxes[row]
        bbox_rec = Rectangle(
            Point(bbox[0], bbox[1]), Point(bbox[0] + bbox[2], bbox[1] + bbox[3])
        )
        if is_rec1_in_rec2(bbox_rec, cropped_img_coordinates):
            result.append(bbox_rec)
    return result


def recalculate_bounding_boxes(
    rectangles: Sequence[Rectangle], coordinates: Rectangle
) -> List[Rectangle]:
    result = []
    for rec in rectangles:
        top_left_x = rec.top_left.x - coordinates.top_left.x
        top_left_y = rec.top_left.y - coordinates.top_left.y

        bottom_right_x = rec.bottom_right.x - coordinates.top_left.x
        bottom_right_y = rec.bottom_right.y - coordinates.top_left.y

        result.append(
            Rectangle(
                Point(top_left_x, top_left_y), Point(bottom_right_x, bottom_right_y)
            )
        )
    return result


if __name__ == "__main__":
    input_data = Path("./temp/annotations")
    output_data_path = Path("./temp/lmu-extracted-2")

    input_annotations_folder = input_data / "annotations"
    input_imgs_folder = input_data / "imgs"

    output_imgs = output_data_path / "imgs"
    output_annotations = output_data_path / "annotations"

    create_directory(output_data_path, overwrite=True)
    create_directory(output_imgs)
    create_directory(output_annotations)

    images = list(input_imgs_folder.iterdir())
    annotations = list(input_annotations_folder.iterdir())
    for counter, image_path in enumerate(images):
        print(f"{counter} of {len(images)}")

        annotation = next(
            input_annotations_folder.glob(f"*{image_path.stem}.txt"), None
        )
        if annotation is None:
            print("Not found annotations for image:", image_path.name)
        else:
            annotation_as_np = annotation_to_numpy(annotation)
            image = cv2.imread(str(image_path))
            contours = extract_obverse_reverse(image, annotation_as_np)
            if contours:
                for i, (cropped_img, new_coordinates) in enumerate(contours):
                    print(f"\t{i} if {len(contours)}")
                    new_image_file_stem = f"{image_path.stem}-{i}"
                    cv2.imwrite(
                        str(output_imgs / f"{new_image_file_stem}.jpg"), cropped_img
                    )

                    bboxes = get_bboxes_belonging_to_crop(
                        annotation_as_np, new_coordinates
                    )
                    bboxes = recalculate_bounding_boxes(bboxes, new_coordinates)
                    BoundingBoxes.from_two_vertices(
                        new_image_file_stem, [bbox.to_list() for bbox in bboxes]
                    ).create_ground_truth_txt(output_annotations)
            else:
                print(
                    f"{image_path.stem} has no contours (e.t. obverse, reverse) to extract"
                )
                shutil.copy(image_path, output_imgs / image_path.name)
                shutil.copy(annotation, output_annotations / annotation.name)
