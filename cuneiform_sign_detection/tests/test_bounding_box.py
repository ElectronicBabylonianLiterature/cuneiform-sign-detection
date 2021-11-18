from cuneiform_sign_detection.bounding_boxes import BoundingBox, BoundingBoxes


def test_bounding_box():
    bounding_box = BoundingBox(top_left_x=0, top_left_y=1, width=2, height=3)
    assert bounding_box.top_left_x == 0
    assert bounding_box.top_left_y == 1
    assert bounding_box.width == 2
    assert bounding_box.height == 3


def test_bounding_box_from_two_vertices():
    bounding_box = BoundingBox.from_two_vertices(*[0, 0, 10, 20])
    assert bounding_box.top_left_x == 0
    assert bounding_box.top_left_y == 0
    assert bounding_box.width == 10
    assert bounding_box.height == 20


def test_bounding_boxes():
    bounding_box = BoundingBox(0, 0, 10, 20)
    bounding_boxes = BoundingBoxes("image_0", [bounding_box])
    assert bounding_boxes.image_id == "image_0"
    assert bounding_boxes.bounding_boxes == [bounding_box]


def test_bounding_boxes_create_ground_truth_txt(tmp_path):
    bounding_box = BoundingBox(0, 0, 10, 20)
    bounding_boxes = BoundingBoxes("image_0", [bounding_box, bounding_box])
    bounding_boxes.create_ground_truth_txt(tmp_path)
    test_path = tmp_path / "gt_image_0.txt"
    assert test_path.read_text() == "0,0,10,20\n0,0,10,20"
