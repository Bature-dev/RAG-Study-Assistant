"""
query_engine.py — Context-Aware RAG Query Engine
==================================================
Uses Gemini for generation, HuggingFace for embeddings.
Supports 6 study modes: chat | practice | review | drill | browse | weak
Auto-detects courses from vector_stores/.
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Local import
from system_prompt import build_prompt

#  CRITICAL FIX: suppress all legacy LangChain global attribute errors
import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = None   # added to fix AttributeError: module 'langchain' has no attribute 'llm_cache'

load_dotenv()

# CONFIGURATION
VECTOR_STORES_DIR = "vector_stores"
SUBTYPES          = ["notes", "exams"]
VALID_MODES       = ["chat", "practice", "review", "drill", "browse", "weak"]

EXAM_KEYWORDS = [
    "past question", "exam question", "past exam", "practice question",
    "sample question", "mid semester", "mid-sem", "test question",
    "examination", "quiz", "give me a question", "past paper",
    "what came out", "likely question", "exam practice", "test me",
    "quiz me", "give me a past"
]

MODE_KEYWORDS = {
    "practice": [
        "test me", "quiz me", "give me a question", "i want to practice",
        "practice mode", "give me a past question", "ask me", "examine me"
    ],
    "review": [
        "explain this", "show me the solution", "i don't understand",
        "walk me through", "check my answer", "review mode", "show answer",
        "what's the answer", "work it out"
    ],
    "drill": [
        "drill me", "drill mode", "topic drill", "give me all questions on",
        "i want to focus on", "5 questions on", "keep asking me"
    ],
    "browse": [
        "what topics", "show me questions from", "what questions appeared",
        "list available", "browse mode", "what do you have on",
        "what's in the database", "show me what"
    ],
    "weak": [
        "what should i focus on", "where am i weak", "what topics haven't i",
        "give me something i struggle with", "weak areas", "my weak points",
        "what should i study", "help me prioritise"
    ]
}


# SETUP — HuggingFace for embeddings, Gemini for generation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print(" ERROR: GOOGLE_API_KEY not found in .env file.")
    print("   Please add: GOOGLE_API_KEY=your_key_here")
    sys.exit(1)

print(" Loading HuggingFace Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(" Initialising web search...")
web_search = DuckDuckGoSearchAPIWrapper(backend="lite")

print(" Initialising Gemini generation model...")
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
    google_api_key=GOOGLE_API_KEY,
    temperature=0.5
)

print(" Engine Ready.\n")


# AUTO-DETECTION UTILITIES
def course_to_store_name(course, subtype):
    base = course.lower().replace(" ", "_")
    return f"faiss_{base}_{subtype}"


def store_name_to_course(folder_name):
    raw = folder_name[len("faiss_"):]
    for subtype in SUBTYPES:
        if raw.endswith(f"_{subtype}"):
            raw = raw[: -(len(subtype) + 1)]
            break
    return raw.replace("_", " ").title()


def get_available_courses():
    """Scans vector_stores/ and returns sorted list of course display names."""
    if not os.path.exists(VECTOR_STORES_DIR):
        return []
    courses = set()
    for folder in os.listdir(VECTOR_STORES_DIR):
        full_path = os.path.join(VECTOR_STORES_DIR, folder)
        if not os.path.isdir(full_path) or not folder.startswith("faiss_"):
            continue
        for subtype in SUBTYPES:
            if folder.endswith(f"_{subtype}"):
                courses.add(store_name_to_course(folder))
                break
    return sorted(courses)


def get_available_subtypes(course):
    return [
        st for st in SUBTYPES
        if os.path.exists(
            os.path.join(VECTOR_STORES_DIR, course_to_store_name(course, st))
        )
    ]


# QUERY ROUTING
def detect_query_type(query):
    if any(kw in query.lower() for kw in EXAM_KEYWORDS):
        return "exams"
    return "notes"


def detect_mode(query, current_mode="chat"):
    query_lower = query.lower()
    for mode, keywords in MODE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return mode
    return current_mode


def load_store(course, subtype):
    folder = course_to_store_name(course, subtype)
    db_path = os.path.join(VECTOR_STORES_DIR, folder)
    if not os.path.exists(db_path):
        return None
    db = FAISS.load_local(
        db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": 4})


def retrieve_context(query, course, query_type):
    primary = query_type
    secondary = "notes" if query_type == "exams" else "exams"

    retriever = load_store(course, primary)
    if retriever:
        return retriever.invoke(query), f"{primary} store"

    retriever = load_store(course, secondary)
    if retriever:
        return retriever.invoke(query), f"{secondary} store (fallback)"

    return [], "no store found"


# MAIN QUERY FUNCTION
def answer_query(query, course, chat_history=None, mode="chat",
                 current_mode="chat"):
    """
    Full RAG pipeline with mode-aware system prompt.

    Returns:
        (answer: str, sources: list, active_mode: str, query_type: str)
    """
    if chat_history is None:
        chat_history = []

    try:
        # A. Resolve active mode
        if mode and mode in VALID_MODES:
            active_mode = mode
        else:
            active_mode = detect_mode(query, current_mode)

        # B. Detect store type
        query_type = detect_query_type(query)
        if active_mode in ("practice", "drill", "browse"):
            query_type = "exams"

        # C. Format conversation history
        recent_history = chat_history[-8:]
        history_text = (
            "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in recent_history
            )
            if recent_history else "No previous conversation."
        )

        # D. Retrieve from vector store
        print(f" Mode: {active_mode} | Store: {query_type}")
        print(f" Searching {course} ({query_type})...")
        local_docs, store_label = retrieve_context(query, course, query_type)

        if not local_docs:
            print(f"  No documents retrieved from {store_label}")

        local_context = "\n\n".join([doc.page_content for doc in local_docs])

        # E. Web search
        print(" Running web search...")
        try:
            web_context = web_search.run(query)
        except Exception as web_err:
            print(f"  Web search failed: {web_err}")
            web_context = "Web search unavailable."

        # F. Build mode-aware prompt
        prompt = build_prompt(
            mode=active_mode,
            course=course,
            context=local_context,
            web_context=web_context,
            history=history_text,
            query=query
        )

        # G. Call Gemini
        print(" Asking Gemini...")
        response = llm.invoke(prompt)

        # H. Package sources
        web_source = {
            "metadata": {"source": "Live Web Search"},
            "page_content": web_context
        }
        all_sources = local_docs + [web_source]

        return response.content, all_sources, active_mode, query_type

    except Exception as e:
        print("\n ENGINE ERROR:")
        print(traceback.format_exc())
        return f"Engine Error: {repr(e)}", [], "chat", "notes"


# TEST BLOCK
if __name__ == "__main__":
    available = get_available_courses()

    if not available:
        print(" No vector stores found. Run ingest.py first.")
        sys.exit(1)

    print(f" {len(available)} course(s) available:")
    for i, c in enumerate(available, 1):
        subtypes = get_available_subtypes(c)
        print(f"  {i}. {c}  [{', '.join(subtypes)}]")

    # Use the first available course for testing
    course = available[0]
    history = []

    test_cases = [
        ("What are the main topics in this course?", "chat"),
        ("Test me on the most important concept.", "practice"),
        ("What topics are covered in past exam papers?", "browse"),
    ]

    print(f"\n Testing: {course}\n")

    for query, expected_mode in test_cases:
        print(f"\n{'='*60}")
        print(f" USER [{expected_mode}]: {query}")
        print("="*60)

        answer, sources, active_mode, qtype = answer_query(
            query=query,
            course=course,
            chat_history=history,
            mode=expected_mode
        )

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        print(f"\n [{active_mode} | {qtype}]:")
        print(answer[:500] + ("..." if len(answer) > 500 else ""))
        print(f" Sources: {len(sources)}")