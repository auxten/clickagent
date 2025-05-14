# ClickAgent RAG System

This repository contains a Retrieval-Augmented Generation (RAG) system that allows you to import text data from various sources (PDF, CSV) and ask questions against the imported data. The system uses vector embeddings to find relevant content and Claude AI to generate answers based on that content.

## Overview

The RAG system consists of several key components:

1. **Vector Database**: Uses [chdb](https://github.com/chdb-io/chdb) (ClickHouse) for efficient storage and similarity search of embeddings.
2. **Embedding Model**: Uses [sentence-transformers](https://www.sbert.net/) with the `multilingual-e5-large` model for generating text embeddings.
3. **QA System**: Uses Claude AI (via Anthropic's API) to generate answers based on retrieved context.
4. **Import Utilities**: Supports importing data from CSV files and PDF documents.

## Requirements

- Python 3.8+
- PyPDF2
- chdb
- pandas
- numpy
- sentence-transformers
- anthropic (Claude API client)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude (required for question answering)

## Usage

### Quick Demo

```bash
python example.py -d bitcoin.db tests/bitcoin.pdf
python example.py -d bitcoin.db -q "What is Bitcoin?"
```
Output:
```
(clickagent) ~/Codes/go/src/github.com/auxten/clickagent (main ✗) ./example.py -d bitcoin.db -q "bitcoin papar author and time"

==================================================
Question: bitcoin papar author and time

Retrieving relevant context...

Relevant context:
- PDF Document (2025-01-01 00:00:00): Bitcoin: A Peer-to-Peer Electronic Cash System
Satoshi Nakamoto
satoshin@gmx.com
www.bitcoin.org
Abstract.
- PDF Document (2025-01-01 00:00:00): Dai, "b-money," http://www.weidai.com/bmoney.txt, 1998.
- PDF Document (2025-01-01 00:00:00): In our case, it is CPU time and electricity that is expended.
- PDF Document (2025-01-01 00:00:00): We assume the sender is an attacker  
who wants to make the recipient believe he paid him for a while, then switch it to pay back to  
himself after some time has passed.

.....

Answer:
Based on the provided context, the author of the Bitcoin whitepaper is Satoshi Nakamoto. The whitepaper is titled "Bitcoin: A Peer-to-Peer Electronic Cash System" and the email address satoshin@gmx.com is listed, along with the website www.bitcoin.org.

However, there is no specific date or time mentioned for when the whitepaper was written or published.
==================================================
```

### Basic Commands

```bash
# Import a CSV file
python example.py tests/vectordb.csv

# Import a PDF file
python example.py tests/bitcoin.pdf

# Ask a question against the existing database
python example.py -q "What is Bitcoin?"

# Use a custom database location
python example.py -d custom_path.db tests/bitcoin.pdf
```

### Command-Line Options

- `-q, --question`: Ask a question against the existing database
- `-d, --database`: Specify a custom database path (default: rag.db)
- `file`: Path to a file to import (CSV or PDF)

### CSV Format

The system expects CSV files in the following format:

```
ID,Sender,SenderName,Content,Timestamp,Duration,Offset
"1","User1","John Doe","This is a sample content.","2023-01-01T12:00:00+00:00","0","0"
```

Required columns:
- `ID`: Unique identifier
- `Sender`: Sender identifier
- `SenderName`: Name of the sender
- `Content`: The actual text content
- `Timestamp`: ISO format timestamp
- `Duration`: Duration in milliseconds
- `Offset`: Offset in milliseconds

### PDF Import

When importing a PDF, the system:
1. Extracts text from the PDF
2. Splits the text into sentences
3. Converts them to a CSV-like format internally
4. Generates embeddings for each sentence
5. Stores the sentences and embeddings in the database

## How RAG Works

1. **Data Ingestion**:
   - Text data is imported from CSV files or PDFs
   - Each text segment (chat message, sentence, etc.) is processed

2. **Embedding Generation**:
   - The multilingual-e5-large model converts text to vector embeddings
   - These embeddings capture the semantic meaning of the text
   - Embeddings are generated in small batches to manage memory usage

3. **Vector Storage**:
   - Text and its corresponding embedding are stored in a ClickHouse database
   - The database uses the MergeTree engine for efficient storage and retrieval

4. **Retrieval Process**:
   - When a question is asked, it's converted to an embedding
   - The system finds the most similar content using cosine similarity
   - ClickHouse's `cosineDistance` function enables efficient similarity search

5. **Answer Generation**:
   - The retrieved similar text segments are used as context
   - Claude AI receives the question and relevant context
   - Claude generates a comprehensive answer based on the provided information

## System Architecture

```
┌─────────────┐    ┌───────────────┐    ┌────────────────┐
│  Input      │    │ Embedding     │    │ Vector         │
│  Sources    ├───►│ Generation    ├───►│ Database       │
│ (CSV, PDF)  │    │ (E5 Model)    │    │ (ClickHouse)   │
└─────────────┘    └───────────────┘    └────────┬───────┘
                                                 │
                          ┌─────────────────────►│
                          │                      │
┌─────────────┐    ┌──────┴──────────┐    ┌──────▼───────┐
│ User        │    │ Answer          │    │ Similarity   │
│ Question    ├───►│ Generation      │◄───┤ Search       │
│             │    │ (Claude AI)     │    │              │
└─────────────┘    └─────────────────┘    └──────────────┘
```

## Example Flow

1. Import data: `python example.py my_document.pdf`
2. Ask a question: `python example.py -q "What are the key features of vector databases?"`
3. The system:
   - Converts the question to an embedding
   - Finds the most similar text segments in the database
   - Sends the question and relevant text to Claude
   - Displays Claude's answer

## Performance Considerations

- For large documents, the import process may take some time due to embedding generation
- The embedding batch size is limited to avoid memory issues
- Vector similarity search is efficient even with large datasets
- Using a local chdb database file allows for persistent storage between sessions

## Troubleshooting

- If you see embedding index errors, try reducing the embedding batch size
- Make sure your ANTHROPIC_API_KEY is properly set if using question answering
- For large PDF files, be patient during the import process
- Database errors might be resolved by deleting the database file and starting fresh

## Extending the System

- Additional data sources can be added by creating new importers
- Different embedding models can be used by modifying the EmbeddingGenerator class
- The system can be integrated with other LLMs besides Claude
- The database schema can be extended to store additional metadata
