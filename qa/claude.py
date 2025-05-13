from typing import List, Dict, Any
import anthropic
from datetime import datetime


class ClaudeQA:
    def __init__(self, api_key: str):
        """
        Initialize Claude QA with API key

        Args:
            api_key: Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key)

    def format_context(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format chat messages into a context string

        Args:
            messages: List of message dictionaries with name, time, and content

        Returns:
            Formatted context string
        """
        context_lines = []
        for msg in messages:
            # Format time if it's a datetime object
            time_str = (
                msg["time"].strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(msg["time"], datetime)
                else msg["time"]
            )
            context_lines.append(f"{msg['name']} ({time_str}): {msg['content']}")

        return "\n".join(context_lines)

    def answer_question(
        self,
        question: str,
        context_messages: List[Dict[str, Any]],
        model: str = "claude-3-sonnet-20240229",
    ) -> str:
        """
        Answer a question based on the provided chat context

        Args:
            question: The question to answer
            context_messages: List of message dictionaries with name, time, and content
            model: Claude model to use

        Returns:
            Answer to the question
        """
        # Format the context
        context = self.format_context(context_messages)

        # Construct the prompt
        prompt = f"""You are a helpful AI assistant. Please answer the question based on the following chat context.
The context is a conversation between people, with each line formatted as "Name (Time): Message".

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context. If the context doesn't contain enough information to answer the question, please say so."""

        # Get response from Claude
        response = self.client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.7,
            system="You are a helpful AI assistant that answers questions based on provided chat context.",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text


if __name__ == "__main__":
    # Example usage
    from store.ch import ChatStore

    # Initialize store and QA
    store = ChatStore(":memory:")
    qa = ClaudeQA("your-api-key-here")

    try:
        # Import test data
        store.import_csv("tests/vectordb.csv")

        # Example question
        question = "请总结一下关于向量数据库的主要观点"

        # Get relevant context using vector search
        context_messages = store.search_similar(question, limit=5)

        # Get answer from Claude
        answer = qa.answer_question(question, context_messages)
        print(f"Question: {question}")
        print("\nAnswer:")
        print(answer)

    finally:
        store.close()
