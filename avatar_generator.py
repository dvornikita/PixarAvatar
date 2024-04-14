import torch
import random
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline
from PIL import Image

from prompt_utils import get_pos_and_neg_prompt


class PixarAvatarGenerator:
    """A class to generate Pixar-style avatars of people using the SDXL model.

    Args:
        generate_prompt_from_image (bool, optional): Whether to generate the prompt based on the image. Defaults to False.
        guidance_scale (int, optional): The scale of the guidance loss. Defaults to 10.
        base_strength (float, optional): The base strength of the diffusion process. Defaults to 0.45.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 40.
        device (str, optional): Device to place the model on. Defaults to "cuda".
    """

    def __init__(
        self,
        generate_prompt_from_image=False,
        guidance_scale=10,
        base_strength=0.45,
        num_inference_steps=40,
        device="cuda",
    ):
        self.pipe = self.init_model(device)
        self.generate_prompt_from_image = generate_prompt_from_image
        self.gudance_scale = guidance_scale
        self.base_strength = base_strength
        self.num_inference_steps = num_inference_steps

    def init_model(self, device):
        """Initializes the SDXL model with the LoRA layers and FreeU diffusion process enhancement."""
        # load the base diffusion model
        model_path = "https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated/blob/main/topRatedTurboxlLCM_v10.safetensors"
        if "cuda" in device:
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                model_path, torch_dtype=torch.float16
            )
            pipe.to("cuda")
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(model_path)

        # load the LoRA layers finetuned for pixar-style images
        pipe.load_lora_weights(
            "ntc-ai/SDXL-LoRA-slider.pixar-style",
            weight_name="pixar-style.safetensors",
            adapter_name="pixar-style",
        )
        pipe.set_adapters(["pixar-style"], adapter_weights=[3.0])
        # enable FreeU diffusion process enhancement
        pipe.enable_freeu(s1=1.2, s2=1.5, b1=1.1, b2=1.2)
        # Define a faster scheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        return pipe

    def generate_prompts(self, image: Image):
        """Generates positive and negative prompts for the diffusion process."""
        if self.generate_prompt_from_image:
            return get_pos_and_neg_prompt(image)
        else:
            return get_pos_and_neg_prompt()

    def generate_avatar(self, image: Image) -> Image:
        """Generates a single Pixar-style avatar of a person based on the provided image."""
        prompt, negative_prompt = self.generate_prompts(image)
        return self.pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            strength=self.base_strength,
            guidance_scale=self.gudance_scale,
            num_inference_steps=self.num_inference_steps,
        ).images[0]

    def generate_multiple_avatars(self, image, num_avatars=4):
        """Generates multiple Pixar-style avatars of a person based on the provided image."""
        prompt, negative_prompt = self.generate_prompts(image)
        images = []
        for _ in range(num_avatars):
            # slightly vary the strength of the diffusion process to get more diverse results
            strength = random.uniform(
                self.base_strength - 0.07, self.base_strength + 0.07
            )
            images.append(
                self.pipe(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    guidance_scale=self.gudance_scale,
                    num_inference_steps=self.num_inference_steps,
                ).images[0]
            )
        return images
