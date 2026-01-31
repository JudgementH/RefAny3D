import math

from PIL import Image


def create_image_grid(images: list[Image.Image], cols: int) -> Image.Image:
    if not images:
        raise ValueError("Image list cannot be empty.")

    img_width, img_height = images[0].size

    num_images = len(images)
    rows = math.ceil(num_images / cols)

    grid_width = cols * img_width
    grid_height = rows * img_height

    grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

    for i, img in enumerate(images):
        current_col = i % cols
        current_row = i // cols

        paste_x = current_col * img_width
        paste_y = current_row * img_height

        grid_image.paste(img, (paste_x, paste_y))

    return grid_image
