from enum import StrEnum


class AnalysisRunType(StrEnum):
    PARSE = "parse"
    METHOD_EXTRACT = "method_extract"
    REPRO_SCORE = "repro_score"
    QA = "qa"
    EXPERIMENT_GEN = "experiment_gen"
