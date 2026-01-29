import os

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

from modules.features import FeatureExtractorType
from modules.matchers import MatcherType


class ConfigLoader:
    def __init__(self, path):
        self.path = path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="ascii") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping")
        self._data = data
        self._runtime = {}

    def set_runtime(self, key, value):
        self._runtime[key] = value

    def get(self, key, default=None):
        if key in self._runtime:
            return self._runtime[key]
        node = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node if node is not None else default

    def _get_enum(self, key, enum_cls, default_name):
        value = self.get(key, default_name)
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            name = value.strip().upper()
            if name in enum_cls.__members__:
                return enum_cls[name]
        raise ValueError(f"Invalid value for {key}: {value}")

    def feature_extractor(self):
        return self._get_enum(
            "registration.homography.feature_extractor",
            FeatureExtractorType,
            "FAST_BRIEF",
        )

    def matcher(self):
        return self._get_enum(
            "registration.homography.matcher",
            MatcherType,
            "BF",
        )
