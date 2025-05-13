import chdb
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from embedding import EmbeddingGenerator


class ChatStore:
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize ChatStore with chdb connection

        Args:
            db_path: Path to database file or ":memory:" for in-memory database
        """
        self.conn = chdb.connect(db_path)
        self.cursor = self.conn.cursor()
        self.embedding_generator = EmbeddingGenerator()
        self._create_table()

    def _create_table(self):
        """Create the chat messages table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id String,
            name String,
            time DateTime,
            content String,
            embedding Array(Float32),
            duration UInt32,
            offset UInt32
        ) ENGINE = Memory()
        """
        self.cursor.execute(create_table_query)

    def import_csv(self, csv_path: str, batch_size: int = 100):
        """
        Import chat messages from CSV file

        Args:
            csv_path: Path to CSV file
            batch_size: Number of records to process in each batch
        """
        df = pd.read_csv(csv_path)

        # Process in batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]

            # Generate embeddings for the batch
            contents = batch_df["Content"].tolist()
            embeddings = self.embedding_generator.generate_embedding(contents)

            # Prepare values for multi-row insert
            value_rows = []
            for idx, row in batch_df.iterrows():
                # Format datetime and escape single quotes in strings
                time_str = datetime.fromisoformat(
                    row["Timestamp"].replace("Z", "+00:00")
                ).strftime("'%Y-%m-%d %H:%M:%S'")
                content = row["Content"].replace("'", "''")
                name = row["SenderName"].replace("'", "''")
                embedding_str = str(embeddings[idx].tolist())

                value_rows.append(
                    f"('{row['ID']}', '{name}', {time_str}, '{content}', {embedding_str}, {row['Duration']}, {row['Offset']})"
                )

            # Build multi-row insert query
            values_str = ",".join(value_rows)
            query = f"""
            INSERT INTO chat_messages 
            (id, name, time, content, embedding, duration, offset)
            VALUES {values_str}
            """

            self.cursor.execute(query)

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar messages using cosine similarity

        Args:
            query: Search query text
            limit: Maximum number of results to return

        Returns:
            List of similar messages with their similarity scores
        """
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query)
        embedding_str = str(query_embedding.tolist())

        # Search query using cosine similarity
        search_query = f"""
        SELECT 
            id,
            name,
            time,
            content,
            duration,
            offset,
            cosineDistance(embedding, {embedding_str}) as distance
        FROM chat_messages
        ORDER BY distance ASC
        LIMIT {limit}
        """

        self.cursor.execute(search_query)
        results = self.cursor.fetchall()

        # Convert results to list of dictionaries
        return [
            {
                "id": row[0],
                "name": row[1],
                "time": row[2],
                "content": row[3],
                "duration": row[4],
                "offset": row[5],
                "similarity": 1 - row[6],  # Convert distance to similarity
            }
            for row in results
        ]

    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()


if __name__ == "__main__":
    # Example usage
    store = ChatStore(":memory:")  # Use in-memory database

    try:
        # Import CSV file from tests directory
        csv_path = "tests/vectordb.csv"
        print(f"Importing data from {csv_path}...")
        store.import_csv(csv_path)
        print("Import completed successfully!")

        # Test different search queries
        test_queries = [
            "向量数据库",
            "embedding",
            "Pinecone",
            "RAG",
            "multilingual-e5-large",
        ]

        for query in test_queries:
            print(f"\nSearching for: {query}")
            results = store.search_similar(query, limit=3)
            for result in results:
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"Content: {result['content']}")
                print("---")
    finally:
        # Always close the connection
        store.close()
