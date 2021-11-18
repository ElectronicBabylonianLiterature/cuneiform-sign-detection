from pathlib import Path
from typing import Sequence, Optional, List

import attr


@attr.s(auto_attribs=True, frozen=True)
class BoundingBox:
    top_left_x: float
    top_left_y: float
    width: float
    height: float

    @classmethod
    def from_two_vertices(
        csl, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> "BoundingBox":
        """
        from format: [xmin, ymin, xmax, ymax] as used in heidelberg ground truth
        """
        return csl(xmin, ymin, xmax - xmin, ymax - ymin)

    def to_list(self) -> List[float]:
        return [self.top_left_x, self.top_left_y, self.width, self.height]


@attr.s(auto_attribs=True, frozen=True)
class BoundingBoxes:
    image_id: str
    bounding_boxes: Sequence[BoundingBox] = []

    @classmethod
    def from_two_vertices(
        csl, image_id: str, list_of_vertices: Sequence[Sequence[float]]
    ) -> "BoundingBoxes":
        """
        from format: [xmin, ymin, xmax, ymax] as used in heidelberg ground truth
        """
        return csl(
            image_id,
            [BoundingBox.from_two_vertices(*vertices) for vertices in list_of_vertices],
        )

    def create_ground_truth_txt(self, path: Optional[Path] = None) -> None:
        gt_filename = Path(f"gt_{self.image_id}.txt")
        path = path / gt_filename if path else gt_filename

        bounding_boxes_str = []
        for bounding_box in self.bounding_boxes:
            bounding_box_list = list(map(lambda x: str(int(x)), bounding_box.to_list()))
            bounding_boxes_str.append(",".join(bounding_box_list))

        with open(path, "w") as f:
            f.write("\n".join(bounding_boxes_str))
