"""
ingest.py — RAG Study Assistant Ingestion Pipeline
====================================================
Fully dynamic — no hardcoded course names.

Scans raw_data/ for any course folder that contains
notes/ and/or exams/ subfolders, then builds a FAISS
vector store for each one found.

Structure expected:
  raw_data/
  └── <Any Course Name>/
      ├── notes/    →  faiss_<course>_notes
      └── exams/    →  faiss_<course>_exams

Adding a new course is as simple as dropping a new folder
into raw_data/ with notes/ and exams/ subfolders, then
re-running this script. No code changes needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader,
)

load_dotenv()

# CONFIGURATION
RAW_DATA_DIR = "raw_data"
OUTPUT_DIR   = "vector_stores"
SUBTYPES     = ["notes", "exams"]

SUPPORTED_EXTENSIONS = {
    ".pdf", ".pptx", ".ppt", ".docx", ".doc",
    ".jpg", ".jpeg", ".png"
}

CHUNK_SIZE       = 1200
CHUNK_OVERLAP    = 200
MIN_CHUNK_LENGTH = 80

# AUTO-DETECTION
def discover_courses(raw_data_dir):
    """
    Scans raw_data/ and returns course names that have
    at least one recognised subtype folder (notes/ or exams/).
    """
    courses = []

    if not os.path.exists(raw_data_dir):
        print(f" '{raw_data_dir}' folder not found.")
        return courses

    for entry in sorted(os.listdir(raw_data_dir)):
        course_path = os.path.join(raw_data_dir, entry)

        if not os.path.isdir(course_path):
            continue

        has_subtype = any(
            os.path.isdir(os.path.join(course_path, st))
            for st in SUBTYPES
        )

        if has_subtype:
            courses.append(entry)
        else:
            print(f"Skipping '{entry}' — no notes/ or exams/ subfolder.")

    return courses

# FILE LOADER ROUTER
def load_file(filepath):
    """
    Routes each file to the correct Unstructured loader.
    PDFs use fast strategy (text layer, low memory).
    Images use hi_res (OCR required).
    """
    ext = Path(filepath).suffix.lower()

    try:
        if ext == ".pdf":
            loader = UnstructuredPDFLoader(
                filepath,
                mode="elements",
                strategy="fast",
            )
        elif ext in (".jpg", ".jpeg", ".png"):
            loader = UnstructuredImageLoader(
                filepath,
                mode="elements",
                strategy="hi_res",
            )
        elif ext in (".pptx", ".ppt"):
            loader = UnstructuredPowerPointLoader(
                filepath,
                mode="elements",
            )
        elif ext in (".docx", ".doc"):
            loader = UnstructuredWordDocumentLoader(
                filepath,
                mode="elements",
            )
        else:
            loader = UnstructuredFileLoader(
                filepath,
                mode="elements",
            )

        return loader.load()

    except Exception as e:
        print(f"Loader failed for {Path(filepath).name}: {e}")
        return []

# STORE NAME UTILITY
def course_to_store_name(course, subtype):
    base = course.lower().replace(" ", "_")
    return f"faiss_{base}_{subtype}"

# MAIN INGESTION PIPELINE
def create_databases(raw_data_dir=RAW_DATA_DIR, output_dir=OUTPUT_DIR):

    print("INGESTION PIPELINE — AUTO-DETECT MODE")
    

    courses = discover_courses(raw_data_dir)

    if not courses:
        print("No valid course folders found in raw_data/.")
        return

    print(f"\n Detected {len(courses)} course(s):")
    for c in courses:
        print(f"   • {c}")
    print()

    print(" Loading HuggingFace Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model ready.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    os.makedirs(output_dir, exist_ok=True)

    total_stores_built   = 0
    total_stores_merged  = 0
    total_files_indexed  = 0
    total_files_failed   = 0
    total_chunks_dropped = 0

    for course in courses:
        for subtype in SUBTYPES:

            src_path = os.path.join(raw_data_dir, course, subtype)

            if not os.path.exists(src_path):
                continue

            db_name  = course_to_store_name(course, subtype)
            db_path  = os.path.join(output_dir, db_name)
            log_path = os.path.join(db_path, ".indexed_files.txt")

            print("=" * 60)
            print(f" {course} / {subtype.upper()}")
            print(f"   Store : {db_name}")
            print("=" * 60)

            #Load indexed file log
            indexed_files = set()
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    indexed_files = set(
                        line.strip() for line in f if line.strip()
                    )

            #Load existing vector store
            existing_db = None
            if os.path.exists(db_path):
                try:
                    existing_db = FAISS.load_local(
                        db_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print(f"Loaded existing store")
                except Exception as e:
                    print(f"Could not load existing store: {e}")

            #Process files
            new_docs      = []
            newly_indexed = []
            success_count = 0
            fail_count    = 0
            skip_count    = 0

            for filename in sorted(os.listdir(src_path)):

                if filename.startswith("~$"):
                    continue

                ext = Path(filename).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                if filename in indexed_files:
                    print(f"  Already indexed: {filename}")
                    skip_count += 1
                    continue

                filepath = os.path.join(src_path, filename)
                print(f"\n  Loading: {filename}")

                docs = load_file(filepath)

                if docs:
                    for doc in docs:
                        doc.metadata["source"] = filename
                        doc.metadata["course"] = course
                        doc.metadata["type"]   = subtype

                    new_docs.extend(docs)
                    newly_indexed.append(filename)
                    print(f"  {len(docs)} elements extracted")
                    success_count += 1
                else:
                    print(f"  No content extracted")
                    fail_count += 1

            total_files_failed += fail_count

            if not new_docs:
                if skip_count > 0:
                    print(f"\n  All files already indexed.\n")
                else:
                    print(f"\n  No supported files found.\n")
                continue

            print(f"\n  Loaded: {success_count} | Failed: {fail_count} | Skipped: {skip_count}")

            #  Chunk 
            print(f"  Chunking {len(new_docs)} elements...")
            chunks = text_splitter.split_documents(new_docs)

            all_chunks   = len(chunks)
            valid_chunks = [
                c for c in chunks
                if isinstance(c.page_content, str)
                and len(c.page_content.strip()) >= MIN_CHUNK_LENGTH
            ]
            dropped = all_chunks - len(valid_chunks)
            total_chunks_dropped += dropped
            print(f" {len(valid_chunks)} valid chunks ({dropped} thin chunks dropped)")

            if not valid_chunks:
                print(f"  No valid chunks after filtering. Skipping.\n")
                continue

            #  Merge or create
            if existing_db is not None:
                print(f" Merging into existing store...")
                new_db = FAISS.from_documents(valid_chunks, embeddings)
                existing_db.merge_from(new_db)
                existing_db.save_local(db_path)
                print(f"  Merged → {db_path}")
                total_stores_merged += 1
            else:
                print(f"  Creating new store...")
                new_db = FAISS.from_documents(valid_chunks, embeddings)
                os.makedirs(db_path, exist_ok=True)
                new_db.save_local(db_path)
                print(f"    Saved → {db_path}")
                total_stores_built += 1

            #  Update log
            with open(log_path, "a", encoding="utf-8") as f:
                for fname in newly_indexed:
                    f.write(fname + "\n")

            total_files_indexed += success_count

    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print("=" * 60)
    print(f"   Courses detected   : {len(courses)}")
    print(f"   Stores created     : {total_stores_built}")
    print(f"   Stores merged      : {total_stores_merged}")
    print(f"   Files indexed      : {total_files_indexed}")
    print(f"   Files failed       : {total_files_failed}")
    print(f"   Thin chunks dropped: {total_chunks_dropped}")
    print(f"   Output dir         : {output_dir}/\n")

# RUN
if __name__ == "__main__":
    create_databases()