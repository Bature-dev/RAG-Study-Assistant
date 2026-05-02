"""
evaluate.py — RAG System Evaluation Script

Embeddings : HuggingFace all-MiniLM-L6-v2
Scoring    : Groq llama-3.1-8b-instant

Setup:
  .env must contain:
    GOOGLE_API_KEY= gemini_key

Resume mode:
  - Cases only marked complete when LLM calls succeed
  - Failed cases retried on next run
  - Resume file kept permanently — never auto-deleted
  - Run: python evaluate.py --status  to see progress
  - Delete evaluation_results/resume_progress.json to start fresh
"""

import os
import re
import sys
import time
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

# CONFIGURATION
VECTOR_STORES_DIR = "vector_stores"
RESULTS_DIR       = "evaluation_results"
SUBTYPES          = ["notes", "exams"]
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
RESUME_FILE = os.path.join(RESULTS_DIR, "resume_progress.json")

GROQ_MODEL = "llama-3.1-8b-instant"
CALL_DELAY = 1
MAX_RETRY  = 3
BASE_WAIT  = 30


# CUSTOM TEST CASES
CUSTOM_TEST_CASES = {

    "Cosc 402 (Information Theory)": {
        "notes": [
            {
                "query": "What is entropy in information theory?",
                "expected_keywords": ["entropy", "uncertainty", "probability", "Shannon", "information"]
            },
            {
                "query": "Explain mutual information and when it is used.",
                "expected_keywords": ["mutual information", "joint", "probability", "entropy"]
            },
            {
                "query": "What is the channel capacity theorem?",
                "expected_keywords": ["capacity", "channel", "Shannon", "bandwidth", "noise"]
            },
            {
                "query": "Define joint entropy and conditional entropy.",
                "expected_keywords": ["joint entropy", "conditional", "entropy", "probability"]
            },
        ],
        "exams": [
            {
                "query": "Give me a past exam question on Shannon's theorem.",
                "expected_keywords": ["Shannon", "entropy", "theorem", "capacity", "information"]
            },
            {
                "query": "What types of questions appear in COSC 402 exams?",
                "expected_keywords": ["entropy", "probability", "channel", "information"]
            },
        ],
    },

    "Cosc 423 (Artificial Intelligence)": {
        "notes": [
            {
                "query": "What is an intelligent agent in AI?",
                "expected_keywords": ["agent", "environment", "percept", "action", "rational"]
            },
            {
                "query": "Explain the difference between informed and uninformed search.",
                "expected_keywords": ["heuristic", "BFS", "DFS", "informed", "uninformed"]
            },
            {
                "query": "What are the key characteristics of fifth generation computers?",
                "expected_keywords": ["AI", "parallel", "natural language", "knowledge", "expert"]
            },
            {
                "query": "Who is Edward Feigenbaum and what is his contribution to AI?",
                "expected_keywords": ["Feigenbaum", "expert system", "knowledge", "AI"]
            },
        ],
        "exams": [
            {
                "query": "Give me a past exam question on intelligent agents.",
                "expected_keywords": ["agent", "percept", "environment", "rational", "action"]
            },
            {
                "query": "What topics appeared in the COSC 423 exam?",
                "expected_keywords": ["search", "agent", "AI", "knowledge", "expert"]
            },
        ],
    },

    "Cosc 430 (Hands On Java)": {
        "notes": [
            {
                "query": "What is the difference between an interface and an abstract class in Java?",
                "expected_keywords": ["interface", "abstract", "implement", "extend", "method"]
            },
            {
                "query": "Explain exception handling in Java.",
                "expected_keywords": ["try", "catch", "finally", "throw", "exception"]
            },
            {
                "query": "What are Java collections and why are they useful?",
                "expected_keywords": ["collection", "list", "map", "set", "iterator"]
            },
            {
                "query": "Explain inheritance and polymorphism in Java.",
                "expected_keywords": ["inheritance", "polymorphism", "extends", "override", "class"]
            },
        ],
        "exams": [
            {
                "query": "Give me a past exam question on Java exception handling.",
                "expected_keywords": ["exception", "try", "catch", "throw", "error"]
            },
            {
                "query": "What Java topics appear most in past exam questions?",
                "expected_keywords": ["class", "method", "object", "interface", "inheritance"]
            },
        ],
    },

    "Itgy 307 (Linux Fundamentals)": {
        "notes": [
            {
                "query": "What are the basic Linux file system commands?",
                "expected_keywords": ["ls", "cd", "mkdir", "rm", "chmod"]
            },
            {
                "query": "How do you manage users and groups in Linux?",
                "expected_keywords": ["useradd", "groupadd", "passwd", "sudo", "permissions"]
            },
            {
                "query": "What is the Linux file permission system?",
                "expected_keywords": ["read", "write", "execute", "chmod", "owner"]
            },
            {
                "query": "Explain the Linux process management commands.",
                "expected_keywords": ["ps", "kill", "top", "process", "pid"]
            },
        ],
        "exams": [
            {
                "query": "Give me a past exam question on Linux commands.",
                "expected_keywords": ["command", "linux", "file", "directory", "permission"]
            },
            {
                "query": "What Linux topics appear in past exam questions?",
                "expected_keywords": ["linux", "command", "file", "user", "process"]
            },
        ],
    },

    "Seng 412 (Internet Technology Php)": {
        "notes": [
            {
                "query": "What is PHP and what is it used for?",
                "expected_keywords": ["PHP", "server", "web", "script", "dynamic"]
            },
            {
                "query": "Explain how HTTP works in web communication.",
                "expected_keywords": ["HTTP", "request", "response", "client", "server"]
            },
            {
                "query": "How does PHP handle form data from HTML?",
                "expected_keywords": ["POST", "GET", "form", "input", "superglobal"]
            },
            {
                "query": "What is the difference between GET and POST in PHP?",
                "expected_keywords": ["GET", "POST", "request", "form", "method"]
            },
        ],
        "exams": [
            {
                "query": "Give me a past exam question on PHP basics.",
                "expected_keywords": ["PHP", "variable", "function", "array", "output"]
            },
            {
                "query": "What web development topics appear in SENG 412 exams?",
                "expected_keywords": ["PHP", "HTTP", "HTML", "web", "server"]
            },
        ],
    },
}

FALLBACK_TEST_CASES = {
    "notes": [
        {"query": "What are the main topics covered in this course?",   "expected_keywords": []},
        {"query": "Explain the most important concept in this course.", "expected_keywords": []},
    ],
    "exams": [
        {"query": "Give me a sample past exam question from this course.", "expected_keywords": []},
    ],
}


# SETUP
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")  # not used for embeddings
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print(" GROQ_API_KEY not found in .env file.")
    print(" Get one free at: console.groq.com → API Keys")
    sys.exit(1)

print(" Loading HuggingFace Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

groq_client = Groq(api_key=GROQ_API_KEY)

print(f" Models ready.")
print(f"   Embeddings : HuggingFace all-MiniLM-L6-v2")
print(f"   Evaluator  : {GROQ_MODEL} via Groq (free)\n")


# AUTO-DETECTION
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


def build_test_cases(courses):
    test_cases = []
    test_id    = 1
    for course in courses:
        available_subtypes = get_available_subtypes(course)
        case_bank = CUSTOM_TEST_CASES.get(course, FALLBACK_TEST_CASES)
        for subtype in available_subtypes:
            cases_for_type = case_bank.get(
                subtype, FALLBACK_TEST_CASES.get(subtype, [])
            )
            for case in cases_for_type:
                test_cases.append({
                    "test_id":           test_id,
                    "query":             case["query"],
                    "course":            course,
                    "store_type":        subtype,
                    "expected_keywords": case.get("expected_keywords", [])
                })
                test_id += 1
    return test_cases


# RESUME MODE
def load_progress():
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "completed_ids" in data and "complete_ids" not in data:
                    data["complete_ids"]   = data.pop("completed_ids", [])
                    data["incomplete_ids"] = []
                return data
        except Exception:
            pass
    return {"complete_ids": [], "incomplete_ids": [], "results": []}


def save_progress(complete_ids, incomplete_ids, results):
    with open(RESUME_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "complete_ids":   list(complete_ids),
            "incomplete_ids": list(incomplete_ids),
            "results":        results,
            "last_updated":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)


def is_case_complete(result):
    has_latency      = result.get("latency_seconds")  is not None
    has_relevance    = result.get("answer_relevance") is not None
    has_faithfulness = result.get("faithfulness")     is not None
    return has_latency and (has_relevance or has_faithfulness)


# STATUS REPORT
def print_status_report(test_cases):
    progress       = load_progress()
    complete_ids   = set(progress.get("complete_ids", []))
    incomplete_ids = set(progress.get("incomplete_ids", []))
    results_map    = {r["test_id"]: r for r in progress.get("results", [])}

    print("\n" + "=" * 65)
    print(" EVALUATION STATUS REPORT")
    print("=" * 65)

    not_started = []
    for case in test_cases:
        i = case["test_id"]
        if i in complete_ids:
            r   = results_map.get(i, {})
            lat = f"{r.get('latency_seconds', 0):.2f}s"
            rel = f"{r.get('answer_relevance', '?')}/5"
            fai = f"{r.get('faithfulness', '?')}/5"
            prc = f"{r.get('precision_at_3', 0):.2f}"
            print(f"  [{i:02d}] {case['course'][:28]:<28} "
                  f"({case['store_type']:<5}) "
                  f"P={prc} L={lat} R={rel} F={fai}")
        elif i in incomplete_ids:
            print(f"  [{i:02d}] {case['course'][:28]:<28} "
                  f"({case['store_type']:<5}) FAILED — will retry")
        else:
            not_started.append(i)
            print(f"   [{i:02d}] {case['course'][:28]:<28} "
                  f"({case['store_type']:<5}) NOT YET RUN")

    print(f"\n  ✅ Complete   : {len(complete_ids)}")
    print(f"  ❌ Failed     : {len(incomplete_ids)}")
    print(f"  ⏳ Not started: {len(not_started)}")
    print(f"  📊 Total      : {len(test_cases)}")
    print("=" * 65 + "\n")


# GROQ LLM CALLER
def call_llm(prompt):
    time.sleep(CALL_DELAY)

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.0,
            )
            return response.choices[0].message.content

        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                wait = 30 * attempt
                print(f"      Rate limited (attempt {attempt}/{MAX_RETRY}). "
                      f"Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRY:
                wait = 5 * attempt
                print(f"       Error (attempt {attempt}/{MAX_RETRY}): "
                      f"{str(e)[:60]}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Groq call failed after {MAX_RETRY} retries.")


def parse_json_response(raw_text):
    clean = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*"score"[^{}]*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    score_match  = re.search(r'"score"\s*:\s*(\d)', clean)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', clean)
    if score_match:
        return {
            "score":  int(score_match.group(1)),
            "reason": reason_match.group(1) if reason_match else "Extracted from response"
        }
    raise ValueError(f"Could not parse JSON from: {raw_text[:120]}")


# RETRIEVER
def load_retriever(course, store_type):
    folder  = course_to_store_name(course, store_type)
    db_path = os.path.join(VECTOR_STORES_DIR, folder)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No store for '{course}' ({store_type})")
    db = FAISS.load_local(
        db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": 3})


# METRIC 1 — RETRIEVAL PRECISION@3
def evaluate_retrieval_precision(query, course, store_type, expected_keywords):
    try:
        retriever = load_retriever(course, store_type)
        docs      = retriever.invoke(query)

        if not docs:
            return 0.0, [], "No documents retrieved"

        if not expected_keywords:
            return 1.0, [
                {
                    "chunk_index":      i + 1,
                    "is_relevant":      True,
                    "matched_keywords": ["(no keywords)"],
                    "source":           d.metadata.get("source", "unknown")
                }
                for i, d in enumerate(docs)
            ], "OK (no keywords)"

        relevant_count = 0
        chunk_details  = []
        for i, doc in enumerate(docs):
            content_lower    = doc.page_content.lower()
            matched_keywords = [
                kw for kw in expected_keywords
                if kw.lower() in content_lower
            ]
            is_relevant = len(matched_keywords) > 0
            if is_relevant:
                relevant_count += 1
            chunk_details.append({
                "chunk_index":      i + 1,
                "is_relevant":      is_relevant,
                "matched_keywords": matched_keywords,
                "source":           doc.metadata.get("source", "unknown"),
            })

        return relevant_count / len(docs), chunk_details, "OK"

    except FileNotFoundError as e:
        return None, [], str(e)
    except Exception as e:
        return None, [], f"Retrieval error: {repr(e)}"


# METRIC 2 — RESPONSE LATENCY
def measure_latency(query, course, store_type):
    try:
        retriever = load_retriever(course, store_type)
        start     = time.perf_counter()
        docs      = retriever.invoke(query)
        context   = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            f"You are an academic tutor for {course}. "
            f"Answer this question using the context below.\n\n"
            f"Context: {context[:1500]}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely in 2-3 sentences."
        )

        answer  = call_llm(prompt)
        latency = time.perf_counter() - start
        latency = max(0, latency - CALL_DELAY)

        return latency, answer, "OK"

    except FileNotFoundError as e:
        return None, "", str(e)
    except Exception as e:
        return None, "", f"Latency error: {repr(e)}"


# METRIC 3 — ANSWER RELEVANCE
def evaluate_answer_relevance(query, answer):
    if not answer:
        return None, "No valid answer to evaluate"

    prompt = (
        f"You are an objective evaluator for an AI tutoring system.\n"
        f"Rate how well the ANSWER addresses the QUESTION on a scale of 1 to 5.\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER: {answer}\n\n"
        f"5 = Fully and clearly answers the question\n"
        f"4 = Mostly answers with minor gaps\n"
        f"3 = Partially answers, missing key points\n"
        f"2 = Tangentially related, does not answer\n"
        f"1 = Completely irrelevant or wrong\n\n"
        f"Respond ONLY with valid JSON. Example: "
        f'{{"score": 4, "reason": "The answer covers most aspects clearly."}}\n\n'
        f"Your JSON response:"
    )

    try:
        raw    = call_llm(prompt)
        result = parse_json_response(raw)
        score  = int(result.get("score", 0))
        reason = result.get("reason", "")
        if not 1 <= score <= 5:
            return None, f"Score out of range: {score}"
        return score, reason
    except Exception as e:
        return None, f"Relevance eval error: {repr(e)}"


# METRIC 4 — FAITHFULNESS
def evaluate_faithfulness(query, answer, retrieved_context):
    if not answer or not retrieved_context:
        return None, "Missing answer or context"

    prompt = (
        f"You are an objective evaluator for an AI tutoring system.\n"
        f"Rate how faithfully the ANSWER is grounded in the CONTEXT "
        f"on a scale of 1 to 5.\n\n"
        f"QUESTION: {query}\n\n"
        f"CONTEXT: {retrieved_context[:1500]}\n\n"
        f"ANSWER: {answer}\n\n"
        f"5 = Every claim directly supported by context\n"
        f"4 = Most claims supported, minor additions\n"
        f"3 = About half grounded in context\n"
        f"2 = Most goes beyond provided context\n"
        f"1 = Entirely unsupported or contradicts context\n\n"
        f"Respond ONLY with valid JSON. Example: "
        f'{{"score": 5, "reason": "All claims are directly from the context."}}\n\n'
        f"Your JSON response:"
    )

    try:
        raw    = call_llm(prompt)
        result = parse_json_response(raw)
        score  = int(result.get("score", 0))
        reason = result.get("reason", "")
        if not 1 <= score <= 5:
            return None, f"Score out of range: {score}"
        return score, reason
    except Exception as e:
        return None, f"Faithfulness eval error: {repr(e)}"


# MAIN EVALUATION RUNNER
def run_evaluation():

    courses    = get_available_courses()
    test_cases = build_test_cases(courses)

    if not test_cases:
        print(" No test cases. Run ingest.py first.")
        return

    progress       = load_progress()
    complete_ids   = set(progress.get("complete_ids", []))
    incomplete_ids = set(progress.get("incomplete_ids", []))
    all_results    = progress.get("results", [])

    remaining = [
        c for c in test_cases
        if c["test_id"] not in complete_ids
    ]

    print("=" * 60)
    print(" RAG SYSTEM EVALUATION")
    print("=" * 60)
    print(f"  Embeddings          : HuggingFace all-MiniLM-L6-v2")
    print(f"  Evaluator           : {GROQ_MODEL} (Groq free)")
    print(f"  Courses detected    : {len(courses)}")
    for c in courses:
        subtypes = get_available_subtypes(c)
        source   = "custom" if c in CUSTOM_TEST_CASES else "fallback"
        print(f"    • {c}  [{', '.join(subtypes)}]  ({source})")
    print(f"  Total test cases    : {len(test_cases)}")
    print(f"  ✅ Complete         : {len(complete_ids)}")
    print(f"  ❌ Retry (failed)   : {len(incomplete_ids)}")
    print(f"  ⏳ Remaining        : {len(remaining)}")
    print(f"  Timestamp           : {TIMESTAMP}\n")

    if not remaining:
        print("✅ All test cases already complete.")
        finalize(all_results, courses, complete_ids, incomplete_ids)
        return

    for case in remaining:
        i          = case["test_id"]
        query      = case["query"]
        course     = case["course"]
        store_type = case["store_type"]
        keywords   = case["expected_keywords"]

        print(f"\n[{i}/{len(test_cases)}] {course} ({store_type})")
        print(f"  Q: {query[:65]}...")
        print("-" * 60)

        result = {
            "test_id":             i,
            "course":              course,
            "store_type":          store_type,
            "query":               query,
            "expected_keywords":   ", ".join(keywords),
            "precision_at_3":      None,
            "latency_seconds":     None,
            "answer_relevance":    None,
            "faithfulness":        None,
            "answer_preview":      "",
            "retrieved_context":   "",
            "precision_detail":    "",
            "relevance_reason":    "",
            "faithfulness_reason": "",
            "errors":              ""
        }

        errors = []
        answer = ""

        #  Metric 1 
        print("  Retrieval Precision@3...")
        precision, chunk_details, p_status = evaluate_retrieval_precision(
            query, course, store_type, keywords
        )
        if precision is not None:
            result["precision_at_3"]   = round(precision, 3)
            result["precision_detail"] = " | ".join([
                f"Chunk {c['chunk_index']}: "
                f"{'✅' if c['is_relevant'] else '❌'} "
                f"[{', '.join(c['matched_keywords']) or 'no match'}] "
                f"src={c['source']}"
                for c in chunk_details
            ])
            print(f"     → {precision:.3f} ({int(precision*3)}/3 relevant)")
        else:
            errors.append(f"Precision: {p_status}")
            print(f"     Warinng!  {p_status}")

        #  Metric 2 
        print("    Response Latency...")
        latency, answer, l_status = measure_latency(query, course, store_type)
        if latency is not None:
            result["latency_seconds"] = round(latency, 3)
            result["answer_preview"]  = answer[:300].replace("\n", " ")
            print(f"     → {latency:.3f}s")
        else:
            errors.append(f"Latency: {l_status}")
            print(f"     Warning!  {l_status}")

        try:
            retriever = load_retriever(course, store_type)
            docs      = retriever.invoke(query)
            result["retrieved_context"] = "\n\n".join(
                [d.page_content for d in docs]
            )
        except Exception:
            result["retrieved_context"] = ""

        #  Metric 3 
        print("   Answer Relevance...")
        relevance_score, relevance_reason = evaluate_answer_relevance(
            query, answer
        )
        if relevance_score is not None:
            result["answer_relevance"] = relevance_score
            result["relevance_reason"] = relevance_reason
            print(f"     → {relevance_score}/5 — {relevance_reason[:65]}")
        else:
            errors.append(f"Relevance: {relevance_reason}")
            print(f"     Warning!  {relevance_reason}")

        #  Metric 4 
        print("   Faithfulness...")
        faith_score, faith_reason = evaluate_faithfulness(
            query, answer, result["retrieved_context"]
        )
        if faith_score is not None:
            result["faithfulness"]        = faith_score
            result["faithfulness_reason"] = faith_reason
            print(f"     → {faith_score}/5 — {faith_reason[:65]}")
        else:
            errors.append(f"Faithfulness: {faith_reason}")
            print(f"     Warning!  {faith_reason}")

        result["errors"] = "; ".join(errors)

        if is_case_complete(result):
            incomplete_ids.discard(i)
            complete_ids.add(i)
            all_results = [r for r in all_results if r["test_id"] != i]
            all_results.append(result)
            print(f"   ✅ Case {i} COMPLETE "
                  f"({len(complete_ids)}/{len(test_cases)} done)")
        else:
            incomplete_ids.add(i)
            complete_ids.discard(i)
            print(f"   ❌ Case {i} INCOMPLETE — will retry next run")

        save_progress(complete_ids, incomplete_ids, all_results)

    finalize(all_results, courses, complete_ids, incomplete_ids)


# FINALIZE
def finalize(all_results, courses, complete_ids, incomplete_ids):
    if not all_results:
        print("\n Warning!!  No completed results to save yet.")
        return

    save_detailed_results(all_results)
    save_summary_results(all_results, courses)
    generate_human_scoring_sheet(all_results)
    print_console_summary(all_results, courses)

    if incomplete_ids:
        print(f"\n Warning!!  {len(incomplete_ids)} case(s) still incomplete.")
        print(f"   Run again to retry.")
    else:
        print(f"\n🎉 All {len(complete_ids)} cases complete!")

    print(f"\n Resume file: {RESUME_FILE}")
    print(f"   Delete it to start completely fresh.\n")


def save_detailed_results(results):
    filepath = os.path.join(
        RESULTS_DIR, f"evaluation_results_{TIMESTAMP}.csv"
    )
    fieldnames = [
        "test_id", "course", "store_type", "query", "expected_keywords",
        "precision_at_3", "latency_seconds", "answer_relevance", "faithfulness",
        "answer_preview", "precision_detail",
        "relevance_reason", "faithfulness_reason", "errors"
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n Detailed results    → {filepath}")


def save_summary_results(results, courses):
    filepath = os.path.join(
        RESULTS_DIR, f"evaluation_summary_{TIMESTAMP}.csv"
    )
    groups = {}
    for r in results:
        key = (r["course"], r["store_type"])
        if key not in groups:
            groups[key] = {
                "precision": [], "latency": [],
                "relevance": [], "faithfulness": []
            }
        if r["precision_at_3"]   is not None: groups[key]["precision"].append(r["precision_at_3"])
        if r["latency_seconds"]  is not None: groups[key]["latency"].append(r["latency_seconds"])
        if r["answer_relevance"] is not None: groups[key]["relevance"].append(r["answer_relevance"])
        if r["faithfulness"]     is not None: groups[key]["faithfulness"].append(r["faithfulness"])

    def avg(lst): return round(sum(lst)/len(lst), 3) if lst else "N/A"

    rows = []
    all_p, all_l, all_r, all_f = [], [], [], []

    for (course, store_type), data in sorted(groups.items()):
        rows.append({
            "course":               course,
            "store_type":           store_type,
            "test_cases":           len([r for r in results if r["course"] == course and r["store_type"] == store_type]),
            "avg_precision_at_3":   avg(data["precision"]),
            "avg_latency_seconds":  avg(data["latency"]),
            "avg_answer_relevance": avg(data["relevance"]),
            "avg_faithfulness":     avg(data["faithfulness"]),
        })
        all_p.extend(data["precision"])
        all_l.extend(data["latency"])
        all_r.extend(data["relevance"])
        all_f.extend(data["faithfulness"])

    rows.append({
        "course": "OVERALL", "store_type": "all",
        "test_cases":           len(results),
        "avg_precision_at_3":   avg(all_p),
        "avg_latency_seconds":  avg(all_l),
        "avg_answer_relevance": avg(all_r),
        "avg_faithfulness":     avg(all_f),
    })

    fieldnames = [
        "course", "store_type", "test_cases",
        "avg_precision_at_3", "avg_latency_seconds",
        "avg_answer_relevance", "avg_faithfulness"
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f" Summary             → {filepath}")


def generate_human_scoring_sheet(results):
    filepath = os.path.join(
        RESULTS_DIR, f"human_scoring_sheet_{TIMESTAMP}.csv"
    )
    fieldnames = [
        "test_id", "course", "store_type", "query", "system_answer",
        "clarity_1_5", "accuracy_1_5", "helpfulness_1_5",
        "completeness_1_5", "overall_1_5", "evaluator_comments"
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "test_id":            r["test_id"],
                "course":             r["course"],
                "store_type":         r["store_type"],
                "query":              r["query"],
                "system_answer":      r["answer_preview"],
                "clarity_1_5":        "",
                "accuracy_1_5":       "",
                "helpfulness_1_5":    "",
                "completeness_1_5":   "",
                "overall_1_5":        "",
                "evaluator_comments": ""
            })
    print(f" Human scoring sheet → {filepath}")
    print(f"   → Share with evaluators. Fill columns F–K.")


def print_console_summary(results, courses):
    valid = lambda lst: [x for x in lst if x is not None]
    def avg(lst): return sum(lst)/len(lst) if lst else 0

    precisions   = valid([r["precision_at_3"]   for r in results])
    latencies    = valid([r["latency_seconds"]  for r in results])
    relevances   = valid([r["answer_relevance"] for r in results])
    faithfulness = valid([r["faithfulness"]     for r in results])

    print("\n")
    print("=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Cases in report            : {len(results)}")
    print(f"  Embeddings                 : HuggingFace all-MiniLM-L6-v2")
    print(f"  Evaluator                  : {GROQ_MODEL}")
    print()
    print(f"  Avg Retrieval Precision@3  : {avg(precisions):.3f}  ({len(precisions)} cases)")
    print(f"  Avg Response Latency       : {avg(latencies):.3f}s ({len(latencies)} cases)")
    print(f"  Avg Answer Relevance       : {avg(relevances):.2f}/5 ({len(relevances)} cases)")
    print(f"  Avg Faithfulness           : {avg(faithfulness):.2f}/5 ({len(faithfulness)} cases)")
    print()

    print(f"  {'Course':<40} {'Store':<8} {'P@3':>6} {'Lat':>7} {'Rel':>6} {'Faith':>6}")
    print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")

    for course in courses:
        for subtype in SUBTYPES:
            sub = [r for r in results
                   if r["course"] == course and r["store_type"] == subtype]
            if not sub:
                continue
            p  = avg(valid([r["precision_at_3"]   for r in sub]))
            l  = avg(valid([r["latency_seconds"]  for r in sub]))
            re = avg(valid([r["answer_relevance"] for r in sub]))
            fa = avg(valid([r["faithfulness"]     for r in sub]))
            print(f"  {course:<40} {subtype:<8} {p:>6.3f} {l:>7.3f} {re:>6.2f} {fa:>6.2f}")

    errors = [r for r in results if r["errors"]]
    if errors:
        print(f"\n  Warning !!  {len(errors)} case(s) had partial errors:")
        for r in errors:
            print(f"     [{r['test_id']}] {r['course']} "
                  f"({r['store_type']}) — {r['errors'][:60]}")

    print(f"\n  Output: {RESULTS_DIR}/")
    print("=" * 60)


# ENTRY POINT
if __name__ == "__main__":
    if "--status" in sys.argv:
        courses    = get_available_courses()
        test_cases = build_test_cases(courses)
        print_status_report(test_cases)
    else:
        run_evaluation()