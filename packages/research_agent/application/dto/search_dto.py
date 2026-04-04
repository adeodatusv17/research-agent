from pydantic import BaseModel


class SearchResultDTO(BaseModel):
    paper_id: str
    score: float
