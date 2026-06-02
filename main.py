"""
CLI Entry Point for Custom RAG Pipeline from Scratch.
Supports building the index, running queries, and entering an interactive REPL shell.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

# Load environment variables before importing any other modules
load_dotenv()

from src.pipeline import build_and_save_index, execute_query

def main():
    parser = argparse.ArgumentParser(
        description="RAG from Scratch: A lightweight custom RAG CLI tool."
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Injest documents in data/, generate embeddings, and build the custom vector store."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a query against the custom RAG pipeline and print the grounded answer."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter an interactive terminal-based chat session."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory containing the text corpus (default: data/)"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/vector_store.json",
        help="Path where the serialized index is saved (default: data/vector_store.json)"
    )
    
    args = parser.parse_args()
    
    # 1. Build Index Phase
    if args.build_index:
        try:
            build_and_save_index(data_dir=args.data_dir, index_path=args.index_path)
            sys.exit(0)
        except Exception as e:
            print(f"\n[-] Error building index: {e}", file=sys.stderr)
            sys.exit(1)
            
    # 2. Run Query Phase
    elif args.query:
        run_query_and_print(args.query, args.index_path)
        sys.exit(0)
        
    # 3. Interactive REPL Phase
    elif args.interactive:
        run_interactive_loop(args.index_path)
        sys.exit(0)
        
    # 4. No arguments passed
    else:
        parser.print_help()
        sys.exit(0)

def run_query_and_print(query: str, index_path: str):
    """
    Executes a query and prints the formatted answer and sources to the terminal.
    """
    try:
        answer, sources = execute_query(query, index_path=index_path)
        
        # Clear print with borders for rich aesthetics
        print("=" * 80)
        print("QUESTION:")
        print(f"  {query}")
        print("-" * 80)
        print("ANSWER:")
        # Indent answer for readability
        print("\n".join(f"  {line}" for line in answer.split("\n")))
        print("-" * 80)
        print("SOURCES:")
        
        if not sources:
            print("  No source documents were retrieved.")
        else:
            for idx, source in enumerate(sources, 1):
                filename = source.get("source", "Unknown")
                score = source.get("score", 0.0)
                # Show first 150 chars of the text snippet
                text_snippet = source.get("text", "").replace("\n", " ").strip()
                snippet_preview = text_snippet[:120] + "..." if len(text_snippet) > 120 else text_snippet
                print(f"  [{idx}] {filename} (similarity: {score:.4f})")
                print(f"      \"{snippet_preview}\"")
                
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n[-] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[-] Error executing query: {e}", file=sys.stderr)
        sys.exit(1)

def run_interactive_loop(index_path: str):
    """
    Launches an interactive loop for prompting and querying.
    """
    if not os.path.exists(index_path):
        print(
            f"[-] Index file not found at '{index_path}'.\n"
            "    Please build the index first using 'python main.py --build-index'.",
            file=sys.stderr
        )
        sys.exit(1)
        
    print("=" * 80)
    print("  Welcome to the RAG from Scratch Interactive Shell!")
    print("  Type your questions below. Type 'exit' or 'quit' to close.")
    print("=" * 80)
    
    while True:
        try:
            user_input = input("\nQuery > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("\nGoodbye!")
                break
                
            run_query_and_print(user_input, index_path)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[-] Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
