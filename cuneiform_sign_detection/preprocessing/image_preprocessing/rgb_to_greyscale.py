from pathlib import Path

from PIL import Image


def convert_img_to_greyscale(image_path: Path, output_path: Path) -> None:
    im = Image.open(image_path).convert("L")
    im.save(output_path / image_path)


def save_to_greyscale(img_path: Path, output_path: Path) -> None:
    if img_path.is_file():
        convert_img_to_greyscale(img_path, output_path)
    elif img_path.is_dir():
        for img in img_path.iterdir():
            convert_img_to_greyscale(img, output_path)
    else:
        raise ValueError("arguments have to be either both directories or both files")


if __name__ == "__main__":
    IMGS_PATH = "../../datasets/complete-contours/imgs"
    OUTPUT_PATH = "../../temp/greyscale-complete-contour/imgs"

    save_to_greyscale(Path(IMGS_PATH), Path(OUTPUT_PATH))
