import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def load_documents(data_dir: str = "data/books") -> List[Document]:
    docs = []
    
    # 1. Check if the folder exists
    if not os.path.isdir(data_dir):
        print(f" Error: Folder not found at {os.path.abspath(data_dir)}")
        return docs

    print(f"🔍 Searching for Alice in: {os.path.abspath(data_dir)}")

    # 2. Walk through the folder
    for root, _, files in os.walk(data_dir):
        for fname in files:
            # We specifically want the Markdown or Text version of Alice
            if fname.lower().endswith((".md", ".txt")):
                path = os.path.join(root, fname)
                try:
                    # TextLoader is the most stable loader for .md files
                    loader = TextLoader(path, encoding='utf-8')
                    loaded_file = loader.load()
                    docs.extend(loaded_file)
                    print(f" Successfully loaded: {fname} ({len(loaded_file)} document object)")
                except Exception as e:
                    print(f" Could not load {fname}: {e}")

    if not docs:
        print("Empty-handed! No .md or .txt files found in the directory.")
        
    return docs