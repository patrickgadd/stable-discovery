from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class PromptGeneratorConfig:
    subjects: List[str]
    actions: List[str]
    suffixes_1: List[str]
    suffixes_2: List[str]
    suffixes_3: List[str]
    steps: List[int]
    num_configs: int
    generator_seed: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def init_from_json_file(cls, json_path):
        with open(json_path, "r") as jf:
            json_str = jf.read()
        return PromptGeneratorConfig.init_from_json(json_str)

    @classmethod
    def init_from_json(cls, json_string: str):
        data = json.loads(json_string)
        return cls(**data)
