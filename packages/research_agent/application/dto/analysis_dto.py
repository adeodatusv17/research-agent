from pydantic import BaseModel


class AnalysisDTO(BaseModel):
    paper_id: str
    status: str
