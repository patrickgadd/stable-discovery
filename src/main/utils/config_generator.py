from main.classes.stable_diffusion_config import StableDiffusionConfig, Prompt
from random import sample
import random
from main.config.core_config import CoreConfig
from typing import List, Optional
from main.classes.prompt_generator_config import PromptGeneratorConfig


class ConfigGenerator:
    @staticmethod
    def generate_configs_v1_a(generator_config: PromptGeneratorConfig) -> List[StableDiffusionConfig]:
        return ConfigGenerator.generate_configs_v1(
            subjects=generator_config.subjects,
            actions=generator_config.actions,
            suffixes_1=generator_config.suffixes_1,
            suffixes_2=generator_config.suffixes_2,
            suffixes_3=generator_config.suffixes_3,
            steps=generator_config.steps,
            num_configs=generator_config.num_configs,
            generator_seed=generator_config.generator_seed
        )

    @staticmethod
    def generate_configs_v1(
            subjects: List[str],
            actions: List[str],
            suffixes_1: List[str],
            suffixes_2: List[str],
            suffixes_3: List[str],
            steps: List[int],
            num_configs: int,
            generator_seed: int
    ) -> List[StableDiffusionConfig]:
        if len(subjects) == 0 or len(actions) == 0 or num_configs < 1:
            raise ValueError("It's required to have both subjects and actions. And to output something (du-uh)")
        
        _suffixes_1: List[Optional[str]] = suffixes_1 + [None]
        _suffixes_2: List[Optional[str]] = suffixes_2 + [None]
        _suffixes_3: List[Optional[str]] = suffixes_3 + [None]
        sd_model: str = CoreConfig.stable_diffusion_model()
        configs: List[StableDiffusionConfig] = []
        random.seed(generator_seed)
        # local_rnd = Random()  # Instantiating locally to avoid polluting whatever one might be doing stochastically elsewhere
        # local_rnd.seed(generator_seed)
        # print(f'{generator_seed = }')
        # print(f'{local_rnd = }')
        # print(f'{local_rnd.seed() = }')

        for prompt_seed in range(num_configs):
            prompt = Prompt(
                subject=sample(subjects, 1)[0],
                action=sample(actions, 1)[0],
                suffix_1=sample(_suffixes_1, 1)[0],
                suffix_2=sample(_suffixes_2, 1)[0],
                suffix_3=sample(_suffixes_3, 1)[0],
            )
            configs.append(StableDiffusionConfig(
                prompt=prompt,
                seed=prompt_seed,
                sd_model_name=sd_model,
                num_steps=sample(steps, 1)[0]
            ))

        return configs


def _main():
    print(f'{CoreConfig.get_data_dir() = }')
    pass


if __name__ == '__main__':
    _main()


