#!/usr/bin/env python3

"""
Example usage of ClickAgent's vector store and Claude QA functionality

Usage:
  python example.py file.csv      # Import CSV file
  python example.py file.pdf      # Import PDF file
  python example.py -q "question" # Ask a question against the existing database
"""

import os
import re
import csv
import sys
import tempfile
import argparse
from pathlib import Path
from store.ch import ChatStore
from qa import ClaudeQA

# PyPDF2 is required for PDF processing
import PyPDF2


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2"""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text


def text_to_sentences(text):
    """Split text into sentences"""
    # Simple sentence splitting by punctuation followed by space
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def save_sentences_to_csv(sentences, output_path):
    """Save sentences to a CSV file in the format expected by ChatStore"""
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "ID",
            "Sender",
            "SenderName",
            "Content",
            "Timestamp",
            "Duration",
            "Offset",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, sentence in enumerate(sentences):
            writer.writerow(
                {
                    "ID": str(i + 1),
                    "Sender": "PDF",
                    "SenderName": "PDF Document",
                    "Content": sentence,
                    "Timestamp": "2025-01-01T00:00:00+00:00",
                    "Duration": "0",
                    "Offset": "0",
                }
            )
    return output_path


def import_pdf_to_store(store, pdf_path):
    """Process a PDF file and import its content into the store"""
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting text into sentences...")
    sentences = text_to_sentences(text)
    print(f"Found {len(sentences)} sentences")

    # Create a temporary CSV file
    temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    temp_csv_path = temp_csv.name
    temp_csv.close()

    print(f"Saving sentences to temporary CSV: {temp_csv_path}")
    save_sentences_to_csv(sentences, temp_csv_path)

    print("Importing sentences into store...")
    try:
        # Use a smaller embedding batch size to avoid memory issues
        store.import_csv(temp_csv_path, embedding_batch_size=8)
        print("PDF import completed!")
    except Exception as e:
        print(f"Error importing PDF: {str(e)}")
        raise
    finally:
        # Clean up the temporary file
        os.unlink(temp_csv_path)


def import_file(store, file_path):
    """Import a file into the store based on its extension"""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return False

    try:
        if file_path.suffix.lower() == ".pdf":
            import_pdf_to_store(store, file_path)
            return True
        elif file_path.suffix.lower() == ".csv":
            print(f"Importing CSV file: {file_path}")
            store.import_csv(file_path, embedding_batch_size=8)
            print("CSV import completed!")
            return True
        else:
            print(f"Error: Unsupported file format: {file_path.suffix}")
            return False
    except Exception as e:
        print(f"Error importing file: {str(e)}")
        return False


def ask_question(store, qa, question):
    """Ask a question and get an answer based on context from the store"""
    print(f"\n{'='*50}")
    print(f"Question: {question}")

    try:
        # Get relevant context using vector search
        print("\nRetrieving relevant context...")
        context_messages = store.search_similar(question, limit=10)

        # Print context for reference
        print("\nRelevant context:")
        for msg in context_messages:
            print(f"- {msg['name']} ({msg['time']}): {msg['content']}")

        # Get answer from Claude
        print("\nGenerating answer...")
        answer = qa.answer_question(question, context_messages)

        print("\nAnswer:")
        print(answer)
        print(f"{'='*50}\n")

        return answer
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        return f"Error: {str(e)}"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="ClickAgent vector store and QA tool")

    # Create mutual exclusive group for different modes
    group = parser.add_mutually_exclusive_group(required=True)

    # Question mode
    group.add_argument(
        "-q", "--question", help="Ask a question to the existing database"
    )

    # File import mode
    group.add_argument("file", nargs="?", help="File to import (CSV or PDF)")

    # Database path option
    parser.add_argument(
        "-d",
        "--database",
        default="rag.db",
        help="Path to database file (default: rag.db)",
    )

    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
    except Exception as e:
        print(f"Error parsing arguments: {str(e)}")
        sys.exit(1)

    # Get database path
    db_path = args.database

    # Initialize store and QA
    try:
        store = ChatStore(db_path)
        qa = ClaudeQA(os.getenv("ANTHROPIC_API_KEY"))

        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Warning: ANTHROPIC_API_KEY environment variable not set")
            print("Question answering will not work without an API key")
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        sys.exit(1)

    try:
        if args.question:
            # Question mode
            ask_question(store, qa, args.question)
        elif args.file:
            # Import mode
            success = import_file(store, args.file)
            if not success:
                sys.exit(1)
        else:
            print(
                "Error: Either a question (-q) or a file to import must be specified."
            )
            sys.exit(1)

    finally:
        store.close()


if __name__ == "__main__":
    main()
