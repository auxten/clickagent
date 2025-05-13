"""
Example usage of ClickAgent's vector store and Claude QA functionality
"""

import os
from store.ch import ChatStore
from qa import ClaudeQA


def main():
    # Initialize store and QA
    store = ChatStore(":memory:")
    qa = ClaudeQA(os.getenv("ANTHROPIC_API_KEY"))

    try:
        # Import test data
        print("Importing test data...")
        store.import_csv("tests/vectordb.csv")
        print("Import completed!")

        # Example questions
        questions = [
            "Please summarize the main points about vector databases",
            "What are the features of Pinecone?",
            "What recommendations are there for choosing embedding models?",
            "How does similarity retrieval work in vector databases?",
        ]

        # Process each question
        for question in questions:
            print(f"\n{'='*50}")
            print(f"Question: {question}")

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

    finally:
        store.close()


if __name__ == "__main__":
    main()
