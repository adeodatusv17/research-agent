from dataclasses import dataclass


@dataclass(slots=True)
class CodeBundle:
    model_code: str
    dataset_code: str
    train_code: str
    config_yaml: str
    requirements_txt: str
