"""Answers a question from the retrieved chunks, as a Pydantic object with citations."""
from pydantic import BaseModel, Field

from rag.chunking import Chunk

MODEL = "claude-haiku-5-5"

SYSTEM = """אתה עונה על שאלות של לקוחות חברת הרכב "אופק רכב".
ענה רק על סמך הקטעים שבתוך <chunk>. אל תשתמש בידע כללי.
לכל טענה בתשובה, הוסף ציטוט: מזהה הקטע וציטוט מדויק מתוכו, מילה במילה.
אם הקטעים לא מספיקים כדי לענות, החזר insufficient_context=true, בלי ציטוטים.
ענה בעברית, במשפטים קצרים."""


class Citation(BaseModel):
    chunk_id: str = Field(description="The chunk id, exactly as it appears in <chunk id=...>")
    quote: str = Field(description="An exact quote from that chunk, word for word")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f'<chunk id="{c.id}">\n{c.text}\n</chunk>' for c in chunks)


def ask(question: str, chunks: list[Chunk], client=None) -> tuple[Answer, object]:
    """Returns the parsed Answer and the token usage. Needs ANTHROPIC_API_KEY in the environment."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{build_context(chunks)}\n\nשאלה: {question}"}],
        output_format=Answer,
    )
    return response.parsed_output, response.usage


def _norm(text: str) -> str:
    return " ".join(text.split())


def verify_citations(answer: Answer, chunks: list[Chunk]) -> list[Citation]:
    by_id = {c.id: _norm(c.text) for c in chunks}
    return [c for c in answer.citations
            if c.chunk_id not in by_id or not _norm(c.quote) or _norm(c.quote) not in by_id[c.chunk_id]]