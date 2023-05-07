import pathlib


class CoreConfig:
    @staticmethod
    def get_data_dir() -> str:
        return str(pathlib.Path(__file__).parent.resolve()) + "/../../../data/"

    @staticmethod
    def stable_diffusion_model() -> str:
        return "stabilityai/stable-diffusion-2-1"

    @staticmethod
    def image_quality() -> int:
        # Pick an integer in the range [1, 100] (the output is ultimately going to be JPG)
        return 80

    @staticmethod
    def get_output_dir() -> str:
        return CoreConfig.get_data_dir() + "outputs/"

    @staticmethod
    def get_generator_configs_dir() -> str:
        return CoreConfig.get_data_dir() + "generator-configs/"
