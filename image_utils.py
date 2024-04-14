from PIL import Image
from diffusers.utils import load_image


def extract_image(url: str, crop: str = "center") -> Image:
    """Reads an image from a URL, potentially cropps it to a square, and returns a PIL image.

    Args:
        url (str): URL of the image.
        crop (str): How to crop the image. One of "center", "top", "bottom", or "none".

    Returns:
        PIL.Image: The image.
    """
    assert crop in ["center", "top", "bottom", "none"]
    image = load_image(url).convert("RGB")
    width, height = image.size
    if crop == "none":
        # resacle image to have the largest side 640px
        width, height = image.size
        rescale_coeff = 640 / max(width, height)
        new_width, new_height = int(width * rescale_coeff), int(height * rescale_coeff)
    else:
        new_width = new_height = min(width, height)
        left, right = (width - new_width) / 2, (width + new_width) / 2
        if crop == "center":
            top, bottom = (height - new_height) / 2, (height + new_height) / 2
        if crop == "top":
            top, bottom = 0, new_height
        elif crop == "bottom":
            top, bottom = height - new_height, height
        image = image.crop((left, top, right, bottom))
        new_width, new_height = 512, 512
    image = image.resize((new_width, new_height))
    return image
