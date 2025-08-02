import json
import os
import pickle
import re
import time
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
# from tqdm import tqdm
import faiss
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


# ===== Client for Q&A embedding model=====
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NGC_API_KEY"),
)
# ===== End of Q&A embedding model client =====

# Define data structures
class QueryData:
    def __init__(self, sample_id, question, query):
        self.sample_id = sample_id
        self.question = question
        self.query = query
        self.embedding = None

    def __str__(self):
        return f"QueryData(ID={self.id})"

    

# Embedding Generation

def generate_database_embeddings(database, embedding_model_name, embedding_cache_file, input_type="passage"):
    if os.path.isfile(embedding_cache_file):
        ## Load
        with open(embedding_cache_file, "rb") as f:
            database = pickle.load(f)

    else:
        for sample in database:
            logger.debug(f"Running embedding with {embedding_model_name} for data id: {sample.sample_id}")

            # Generate embedding from summary content
            if sample.embedding is None:
                sample.embedding = client.embeddings.create(
                    input=sample.question,
                    model=embedding_model_name,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "NONE"},
                ).data[0].embedding
                # print("finish to create embedding")
        # Dump
        with open(embedding_cache_file, "wb") as f:
            pickle.dump(database, f)
        
            
    # print("Embeddings generated for all logs")
    return database

def generate_embeddings(context, embedding_model_name, input_type="passage"):
    embedding_vector = client.embeddings.create(
        input=context,
        model=embedding_model_name,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"},
    ).data[0].embedding

    return embedding_vector


# Create Vector Index
def create_faiss_index(database, embedding_model_name, embedding_cache_file):
    # Added input_type as "passage" for database embeddings
    database = generate_database_embeddings(database, embedding_model_name, embedding_cache_file, input_type="passage")
    embeddings = np.array([sample.embedding for sample in database]).astype("float32")
    
    # Initialize GPU resources for cuVS
    assert faiss.get_num_gpus() > 0, "No GPU found"
    gpu_res = faiss.StandardGpuResources()
    device = 0
    
    # Configure cuVS
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = device
    cfg.useFloat16 = False
    cfg.use_cuvs = True
    
    # Create GPU FAISS index with cuVS (L2 distance)
    index = faiss.GpuIndexFlatL2(gpu_res, embeddings.shape[1], cfg)
    index.add(embeddings)
    logger.info(f"GPU cuVS FAISS index created with {index.ntotal} entries on device {device}")
    return index

# Similarity Search Function
def find_similar(input_question, database, index, embedding_model_name, top_k=3):
    # Generate query embedding
    query_embedding = generate_embeddings(input_question, embedding_model_name, input_type="passage") # use type passage for query embedding results in better accuracy
    query_embedding = np.array(query_embedding).reshape(1, -1)
    # Vector search
    distances, indices = index.search(query_embedding, top_k)

#     # Collect results
#     results = []
#     for i, distance in zip(indices[0], distances[0]):
#         if i >= 0:  # FAISS returns -1 for invalid indices
#             results.append(
#                 {
#                     "sample": database[i],
#                     "vector_distance": float(distance),
#                 }
#             )

#     # Sort by combined score
#     results.sort(key=lambda x: x["vector_distance"])
#     return results[:top_k]
    results = indices[0]
    return results


if __name__ == "__main__":
    
    train_pd = pd.read_csv("train_val.csv")
    test_pd = pd.read_csv("test.csv")
    
    embedding_model_name = "nvidia/nv-embedqa-mistral-7b-v2"
    embedding_cache_file = "train_database_embedqa_mistral_7b.pkl"
    
    database = []
    for i in range(len(train_pd)):
        database.append(
            QueryData(
                sample_id=i,
                question=train_pd.iloc[i]["questions"],
                query=train_pd.iloc[i]["labels"],
            )
        )
        
    ## create database index
    database_index = create_faiss_index(database, embedding_model_name)
    
    input_question = test_pd.iloc[107]["questions"]
    results = find_similar(input_question, database, database_index, embedding_model_name, top_k=3)
    print(results)
