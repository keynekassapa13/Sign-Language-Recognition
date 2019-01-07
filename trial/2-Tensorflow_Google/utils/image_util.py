from typing import List
# Image Downloading
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
# Drawing utils
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps


def display_image(image) -> None:
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig("result_boxed")


def download_image(url: str, width: int = 256, height: int = 256, display=False) -> str:
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = resize_image(image_data, width, height)
    image_rgb = image.convert("RGB")
    image_rgb.save(filename, format="JPEG", quality=90)
    print(f"Image downloaded to {filename}")
    if display:
        display_image(image)
    return filename


def resize_image(image_data, width: int, height: int) -> Image:
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (width, height), Image.ANTIALIAS)
    return pil_image


def draw_bounding_box(image: Image,
                      ymin: int, xmin: int,
                      ymax: int, xmax: int,
                      colour: ImageColor,
                      font: ImageFont,
                      thickness: int = 4,
                      display_str_list: List[str] = []) -> None:

    # Make Bounding Box
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=thickness, fill=colour)

    # If total height of display strings + top of bounding box exceeds top of image, stack below instead.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display str has top/bottom margin of 0.05.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=colour)
        draw.text((left + margin, text_bottom - text_height, margin),
                  display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names: List, scores: List, max_boxes: int = 10, min_score: int = 0):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colours = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            display_str = f"{class_names[i].decode('ascii')}: {int(100 * scores[i])}"
            colour = colours[hash(class_names[i]) % len(colours)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box(
                image_pil,
                ymin, xmin,
                ymax, xmax,
                colour, font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image
