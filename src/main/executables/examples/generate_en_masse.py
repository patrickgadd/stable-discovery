from typing import List
from main.classes.stable_diffusion_config import StableDiffusionConfig
from main.utils.config_generator import ConfigGenerator
from main.utils.image_generator import ImageGenerator
from main.config.core_config import CoreConfig
from main.classes.prompt_generator_config import PromptGeneratorConfig


def _main():
    """
    This will write images to ".../data/outputs/{experiment_name}/*.jpg"
    - along with the utilized prompts in the file "configs.newline_json"
    - - each line should be JSON-compatible, but the file as whole isn't regular JSON
    - - the index of each line corresponds to the name of each output image. I.e. this allows you to reproduce individual results, should you so desire
    """

    experiment_name = "golden-glasses-synthwave-01"  # This is the name of your config under ".../data/generator-configs/{experiment_name}.json"
    generator_config = PromptGeneratorConfig.init_from_json_file(CoreConfig.get_generator_configs_dir() + f'{experiment_name}.json')

    configs: List[StableDiffusionConfig] = ConfigGenerator.generate_configs_v1_a(generator_config)

    ImageGenerator.generate(configs=configs, output_dir_relative_path=experiment_name)


if __name__ == '__main__':
    _main()
