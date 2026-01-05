import yaml
from pathlib import Path

class Config(dict):
    def __init__(self, path):
        super().__init__()
        self.path = str(Path(path))  # keep the path if you want
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Top-level YAML must be a mapping; got {type(data).__name__}"
            )
        self.update(data)  # <-- initialize the dict in place
