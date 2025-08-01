import re
from pathlib import Path
from dotenv import load_dotenv
import os
import pickle
load_dotenv()
from loguru import logger
import faiss
from openai import OpenAI
import numpy as np

from openai import AzureOpenAI



def split_sql_blocks(file_path):
    """
    file_path: path to the `eicu_instruct_benchmark_rag.sql` file
    Read an SQL file and split it into blocks of code.
    Each block contains a DROP TABLE and CREATE TABLE statement for one table.
    """
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by "DROP TABLE IF EXISTS" statements
    # This pattern looks for DROP TABLE at the start of a line
    pattern = r'^DROP TABLE IF EXISTS'
    
    # Find all positions where DROP TABLE statements start
    lines = content.split('\n')
    block_starts = []
    
    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            block_starts.append(i)
    
    # Add the end of file as the last position
    block_starts.append(len(lines))
    
    # Extract blocks
    blocks = []
    for i in range(len(block_starts) - 1):
        start_line = block_starts[i]
        end_line = block_starts[i + 1]
        
        # Join lines for this block
        block_lines = lines[start_line:end_line]
        
        # Remove empty lines at the end of the block
        while block_lines and block_lines[-1].strip() == '':
            block_lines.pop()
        
        if block_lines:
            block = '\n'.join(block_lines)
            blocks.append(block)
    
    return blocks


class TextBlock:
    """Class to store text blocks with their embeddings"""
    def __init__(self, block_id, content):
        self.block_id = block_id
        self.content = content
        self.embedding = None


class FAISSRetriever:
    def __init__(self, api_key, api_version, azure_endpoint, model="text-embedding-3-large"):
        """
        Initialize the retriever with NVIDIA API client
        
        Args:
            api_key: NVIDIA API key (if None, will look for NGC_API_KEY in env)
            model: Embedding model name
        """

        # Create a client instance
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

        self.model = model
        self.blocks = []
        self.index = None

    def generate_embedding(self, text):
        """Generate embedding for a single text using NVIDIA API"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding


    def embed_blocks(self, text_blocks, cache_file=None):
        """
        Embed text blocks and create FAISS index
        
        Args:
            text_blocks: List of text strings
            cache_file: Optional pickle file to cache embeddings
        """
        # Check if we have cached embeddings
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.blocks = pickle.load(f)
        else:
            # Create TextBlock objects and generate embeddings
            self.blocks = []
            logger.info(f"Generating embeddings for {len(text_blocks)} blocks...")
            
            for i, content in enumerate(text_blocks):
                block = TextBlock(block_id=i, content=content)
                logger.info(f"Processing block {i+1}/{len(text_blocks)}")
                
                # Generate embedding
                block.embedding = self.generate_embedding(content)
                self.blocks.append(block)
            
            # Save to cache if specified
            if cache_file:
                logger.info(f"Saving embeddings to {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.blocks, f)
        
        # Create FAISS index
        self._create_index()

    def _create_index(self):
        """Create FAISS index from block embeddings"""
        # Convert embeddings to numpy array
        embeddings = np.array([block.embedding for block in self.blocks]).astype('float32')
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"FAISS index created with {self.index.ntotal} entries")

    def retrieve(self, query, top_k=5):
        """
        Retrieve most similar blocks to query
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with block content, ID, and distance
        """
        if self.index is None:
            raise ValueError("No index created. Call embed_blocks() first.")
        
        # Generate query embedding
        logger.info("Generating query embedding...")
        query_embedding = self.generate_embedding(query)
        query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        logger.info(f"Searching for top {top_k} blocks...")
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                results.append({
                    'block_id': self.blocks[idx].block_id,
                    'content': self.blocks[idx].content,
                    'distance': float(distance)
                })
        
        return results