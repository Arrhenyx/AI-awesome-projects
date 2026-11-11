import os
import re
from typing import TypedDict, Literal

import dotenv
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

dotenv.load_dotenv()

# --- Config ---
MODEL_NAME = "openai/gpt-oss-20b"
MAX_REVISIONS = 3
TARGET_SCORE = 10
MIN_SCORE = 1
MAX_SCORE = 10


# --- State ---
class ContentState(TypedDict, total=False):
    topic: str
    draft: str
    score: int
    feedback: str
    revision_count: int
    history: list[dict]  # optional audit trail


# --- LLM ---
llm = ChatGroq(model=MODEL_NAME, temperature=0.7)


# --- Helpers ---
_SCORE_RE = re.compile(r"score\s*[:\-]\s*(\d+)", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"feedback\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_review(text: str) -> tuple[int, str]:
    """Robustly parse 'Score: N' and 'Feedback: ...' from LLM output."""
    score_match = _SCORE_RE.search(text)
    feedback_match = _FEEDBACK_RE.search(text)

    if not score_match:
        # Fallback: treat missing score as needing revision
        return MIN_SCORE, f"Could not parse score. Raw response: {text[:200]}"

    score = int(score_match.group(1))
    score = max(MIN_SCORE, min(MAX_SCORE, score))  # clamp

    feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided."
    return score, feedback


# --- Nodes ---
def write_draft(state: ContentState) -> dict:
    topic = state["topic"]
    feedback = state.get("feedback", "").strip()
    revision_count = state.get("revision_count", 0) + 1

    prompt = f"Write a short paragraph about: {topic}"
    if feedback:
        prompt += (
            f"\n\nYour previous draft received this feedback — address it carefully:\n"
            f"{feedback}"
        )

    response = llm.invoke(prompt)

    return {
        "draft": response.content.strip(),
        "revision_count": revision_count,
    }


def review_draft(state: ContentState) -> dict:
    prompt = f"""You are an extremely strict editor. Rate the paragraph below from 1-10.

Paragraph:
\"\"\"{state['draft']}\"\"\"

Respond in EXACTLY this format (no extra text):
Score: <number>
Feedback: <one sentence>"""

    response = llm.invoke(prompt)
    score, feedback = _parse_review(response.content)

    return {
        "score": score,
        "feedback": feedback,
        "history": state.get("history", []) + [
            {"draft": state["draft"], "score": score, "feedback": feedback}
        ],
    }


def should_revise(state: ContentState) -> Literal["write_draft", "__end__"]:
    if state["score"] >= TARGET_SCORE:
        return END
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return END
    return "write_draft"


# --- Graph ---
builder = StateGraph(ContentState)
builder.add_node("write_draft", write_draft)
builder.add_node("review_draft", review_draft)

builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "review_draft")
builder.add_conditional_edges(
    "review_draft",
    should_revise,
    {"write_draft": "write_draft", END: END},
)

graph = builder.compile()


# --- Entrypoint ---
def main() -> None:
    result = graph.invoke(
        {
            "topic": "why sleep matters for learning",
            "draft": "",
            "feedback": "",
            "score": 0,
            "revision_count": 0,
            "history": [],
        }
    )

    print(f"Final draft ({result['revision_count']} revision(s)):\n")
    print(result["draft"])
    print(f"\nFinal Score: {result['score']}")
    print(f"Feedback: {result.get('feedback', '')}")


if __name__ == "__main__":
    main()
