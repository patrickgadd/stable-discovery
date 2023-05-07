from diffusers import DiffusionPipeline
import torch
from typing import List
from PIL import Image
from main.classes.stable_diffusion_config import StableDiffusionConfig
from main.config.core_config import CoreConfig
import os


class ImageGenerator:
    @staticmethod
    def generate(
            configs: List[StableDiffusionConfig],
            output_dir_relative_path: str
    ):
        for conf in configs:
            if conf.sd_model_name != CoreConfig.stable_diffusion_model():
                raise ValueError(f'Sorry, not supporting a range of SD-models. Just one at a time')

        pipeline = DiffusionPipeline.from_pretrained(configs[0].sd_model_name, torch_dtype=torch.float16)
        pipeline.to("cuda")
        data_dir = CoreConfig.get_output_dir()
        output_dir = data_dir + output_dir_relative_path + "/"
        os.makedirs(output_dir, exist_ok=False)  # 'exist_ok=False' because there's nothing in this framework to handle aggregation across multiple runs

        for conf_idx, conf in enumerate(configs):
            prompt = conf.prompt.as_single_string()
            steps = conf.num_steps
            seed = conf.seed
            print(f'Generating image with N-steps: {steps}, seed: {seed}, and prompt {prompt}')
            img = pipeline(
                prompt,
                num_inference_steps=steps,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images[0]

            output_path = output_dir + f'{conf_idx}.jpg'
            img = img.resize((int(img.size[0]), int(img.size[1])), resample=Image.LANCZOS)
            img.save(output_path, quality=CoreConfig.image_quality())
            # While it's not strictly necessary to write this (as the random sampling should be properly seeded), it's convenient to have readily available
            with open(output_dir + "configs.newline_json", "a") as jf:
                jf.write(conf.to_json() + "\n")
