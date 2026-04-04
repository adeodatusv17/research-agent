from pydantic import BaseModel


class PaperDTO(BaseModel):
    id: str
    title: str
