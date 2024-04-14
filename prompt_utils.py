from typing import Union, Optional
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def describe_person(image: Image, device: Union[str, torch.device] = "cpu"):
    """Generates a description of a person in the image using a VLM model.

    Args:
        image (Image): The image of a person.
        device (str, optional): Device to place the model on. Defaults to "cpu" to because the model is very large.

    Returns:
        str: A sentence descring the person in the image.
    """
    # define the model
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    if "cuda" in device:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to("cuda:0")
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            low_cpu_mem_usage=True,
        )

    # extract the dialog with the model
    prompt = (
        "[INST] <image>\nList the gender, approximate age category, hair color, skin color, detailed facial expression"
        "and the cloths of a person on the photo. Do not mention any other details about the photo. [/INST]"
    )
    inputs = processor(prompt, image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.decode(output[0], skip_special_tokens=True)

    # extract the person description from the dialog output
    person_description = generated_text.split("[/INST]")[1].strip()
    return person_description


def get_pos_and_neg_prompt(image: Image = None) -> tuple[str, str]:
    """Generates a prompt for generating a Pixar-style image of a person. If the image is provided, the prompt will
    include the description of the person in the image, otherwise it will be generic.

    Args:
        image (Image, optional): The image of a person to base the prompt on. Defaults to None.

    Returns:
        tuple[str, str]: A tuple of the positive and negative prompts.
    """
    if image:
        person_description = describe_person(image)
    else:
        person_description = "A person"
    prompt_postfix = (
        "detailed hair, clear beautiful eyes, sharp image, high quality, pixar-style"
    )
    negative_prompt = (
        "unnatural mouth, slanted eyes, poorly drawn hands, ugly, disgusting, poorly drawn feet, "
        "missing limb, mutated, watermark, oversaturated"
    )
    return f"{person_description}, {prompt_postfix}", negative_prompt
