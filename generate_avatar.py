from typing import Optional
import fire
import os
import torch

from avatar_generator import PixarAvatarGenerator
from image_utils import extract_image


def generate_avatar(
    url: str,
    output_folder: str,
    num_avatars: int,
    crop: str = "center",
    generate_prompt_from_image: bool = False,
    device: Optional[str] = None,
):
    """Generates Pixar-style avatars of people based on an image.

    Args:
        url (str): URL of the image.
        output_folder (str): Folder to save the avatars to.
        num_avatars (int): Number of avatars to generate.
        crop (str, optional): How to crop the image. One of "center", "top", "bottom", or "none". Defaults to "center".
        generate_prompt_from_image (bool, optional): Whether to generate the prompt based on the image. Defaults to False.
        device (Optional[str], optional): Device to place the model on. Defaults to "cuda" if available, otherwise "cpu".
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image = extract_image(url, crop=crop)
    avatar_generator = PixarAvatarGenerator(
        generate_prompt_from_image=generate_prompt_from_image, device=device
    )

    # generate avatars
    if num_avatars == 1:
        avatars = [avatar_generator.generate_avatar(image)]
    else:
        avatars = avatar_generator.generate_multiple_avatars(
            image, num_avatars=num_avatars
        )

    # find the folder in output_folder that has the largest number of avatars_xxxx
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    existing_folder_ids = [
        int(float(folder.split("_")[1]))
        for folder in os.listdir(output_folder)
        if folder.startswith("avatars_")
    ]
    folder_id = 0 if len(existing_folder_ids) == 0 else max(existing_folder_ids) + 1

    output_folder = os.path.join(output_folder, f"avatars_{folder_id:04d}")
    os.makedirs(output_folder)
    for i, avatar in enumerate(avatars):
        avatar.save(f"{output_folder}/avatar_{i}.png")
    print("Avatars saved to", output_folder)


@fire.Fire
def main(
    image_url: str,
    output_dir: str = "output",
    num_avatars: int = 4,
    crop: str = "center",
    generate_prompt_from_image: bool = False,
    device: Optional[str] = None,
):
    generate_avatar(
        image_url, output_dir, num_avatars, crop, device, generate_prompt_from_image
    )
