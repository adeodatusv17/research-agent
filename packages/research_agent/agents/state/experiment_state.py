from typing import TypedDict


class ExperimentState(TypedDict, total=False):
    paper_id: str
    experiment_id: str
    artifact_path: str
    generation_status: str
    error_message: str
    db: object
    paper: object
    analysis: object
    repository_recommendation: dict
    config: dict
    config_yaml: str
    defaults_used: list[dict]
    inferred_fields: list[dict]
    validation: dict
    model_code: str
    dataset_code: str
    train_code: str
    utils_code: str
    requirements_txt: str
    metadata: dict
