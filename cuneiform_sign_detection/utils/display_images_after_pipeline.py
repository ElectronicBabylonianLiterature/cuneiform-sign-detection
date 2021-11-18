from torchvision.transforms import transforms


def display_images_after_pipeline(dataset, sample_size=10) -> None:
    for index, i in enumerate(dataset):  # slice in datasets doesn't work
        # https: // github.com / open - mmlab / mmocr / issues / 501
        img = i["img"].data
        transforms.ToPILImage()(img[0]).save(f"temp/dataloader/{index}.jpg")
        if index == sample_size:
            break
