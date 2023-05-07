from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Prompt:
    subject: str
    action: Optional[str]
    suffix_1: Optional[str]
    suffix_2: Optional[str]
    suffix_3: Optional[str]

    def as_single_string(self) -> str:
        full_str = self.subject
        if self.action is not None:
            full_str = full_str + " " + self.action
        if self.suffix_1 is not None:
            full_str = full_str + ", " + self.suffix_1
        if self.suffix_2 is not None:
            full_str = full_str + ", " + self.suffix_2
        if self.suffix_3 is not None:
            full_str = full_str + ", " + self.suffix_3

        return full_str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def init_from_json(cls, json_string: str):
        data = json.loads(json_string)
        return cls(**data)
