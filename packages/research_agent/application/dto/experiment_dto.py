from pydantic import BaseModel


class ExperimentDTO(BaseModel):
    paper_id: str
    generation_status: str
