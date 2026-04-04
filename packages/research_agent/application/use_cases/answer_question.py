class AnswerQuestionUseCase:
    def execute(self, question: str, paper_id: str | None = None) -> dict:
        return {"question": question, "paper_id": paper_id, "answer": None}
