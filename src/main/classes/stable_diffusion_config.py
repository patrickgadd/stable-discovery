import json
from main.classes.prompt import Prompt
from dataclasses import dataclass, asdict


@dataclass
class StableDiffusionConfig:
    prompt: Prompt
    seed: int
    num_steps: int
    sd_model_name: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=lambda o: o.to_json())

    @staticmethod
    def init_from_json(json_string: str):
        json_dict = json.loads(json_string)
        prompt = Prompt(**json_dict["prompt"])
        return StableDiffusionConfig(
            prompt=prompt,
            seed=json_dict["seed"],
            num_steps=json_dict["num_steps"],
            sd_model_name=json_dict["sd_model_name"]
        )
