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
        self._create_database()
        self._create_table()

    def _create_database(self):
        """Create the database if it doesn't exist"""
        try:
            create_database_query = """
            CREATE DATABASE IF NOT EXISTS db ENGINE = Atomic;
            """
            self.cursor.execute(create_database_query)
        except Exception as e:
            print(f"Warning when creating database: {str(e)}")
            print("Continuing with existing database...")

    def _create_table(self):
        """Create the chat messages table if it doesn't exist"""
        try:
            # Use IF NOT EXISTS to avoid errors when table already exists
            create_table_query = """
            CREATE TABLE IF NOT EXISTS db.chat_messages (
                id String,
                name String,
                time DateTime,
                content String,
                embedding Array(Float32),
                duration UInt32,
                offset UInt32
            ) ENGINE = MergeTree()
            ORDER BY id
            """
            self.cursor.execute(create_table_query)
        except Exception as e:
            print(f"Warning when creating table: {str(e)}")
            print("Continuing with existing table...")

    def import_csv(
        self, csv_path: str, batch_size: int = 50, embedding_batch_size: int = 16
    ):
        """
        Import chat messages from CSV file

        Args:
            csv_path: Path to CSV file
            batch_size: Number of records to process in each batch
            embedding_batch_size: Number of embeddings to generate at once
        """
        df = pd.read_csv(csv_path)

        # Process in batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]

            # Prepare values for multi-row insert
            value_rows = []

            # Process embeddings in smaller batches to avoid memory issues
            for j in range(0, len(batch_df), embedding_batch_size):
                sub_batch_df = batch_df.iloc[j : j + embedding_batch_size]

                # Generate embeddings for the sub-batch
                contents = sub_batch_df["Content"].tolist()
                sub_embeddings = self.embedding_generator.generate_embedding(contents)

                for idx, row in sub_batch_df.iterrows():
                    # Calculate relative index within sub-batch
                    sub_idx = idx - batch_df.index[j]

                    # Format datetime and escape single quotes in strings
                    time_str = datetime.fromisoformat(
                        row["Timestamp"].replace("Z", "+00:00")
                    ).strftime("'%Y-%m-%d %H:%M:%S'")
                    content = row["Content"].replace("'", "''")
                    name = row["SenderName"].replace("'", "''")
                    embedding_str = str(sub_embeddings[sub_idx].tolist())

                    value_rows.append(
                        f"('{row['ID']}', '{name}', {time_str}, '{content}', {embedding_str}, {row['Duration']}, {row['Offset']})"
                    )

            # Build multi-row insert query
            values_str = ",".join(value_rows)
            query = f"""
            INSERT INTO db.chat_messages 
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
        FROM db.chat_messages
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
    import os

    # Check if the database file exists
    if os.path.exists("test_store.db"):
        os.remove("test_store.db")

    store = ChatStore("test_store.db")  # Use in-memory database

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
