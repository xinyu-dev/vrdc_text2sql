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

# Load environment variables from .env file
load_dotenv()



client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

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

def generate_database_embeddings(database, embedding_model_name, embedding_cache_file):
    if os.path.isfile(embedding_cache_file):
        ## Load
        with open(embedding_cache_file, "rb") as f:
            database = pickle.load(f)

    else:
        for sample in database:
            # print("processing data id: ", sample.sample_id)

            # Generate embedding from summary content
            if sample.embedding is None:
                sample.embedding = client.embeddings.create(
                    input=sample.question,
                    model=embedding_model_name,
                    encoding_format="float",
                    extra_body={"input_type": "passage", "truncate": "NONE"},
                ).data[0].embedding
                # print("finish to create embedding")
        # Dump
        with open(embedding_cache_file, "wb") as f:
            pickle.dump(database, f)
        
            
    # print("Embeddings generated for all logs")
    return database

def generate_embeddings(context, embedding_model_name):
    embedding_vector = client.embeddings.create(
        input=context,
        model=embedding_model_name,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"},
    ).data[0].embedding

    return embedding_vector


# Create Vector Index
def create_faiss_index(database, embedding_model_name, embedding_cache_file):
    database = generate_database_embeddings(database, embedding_model_name, embedding_cache_file)
    embeddings = np.array([sample.embedding for sample in database]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    # print(f"FAISS index created with {index.ntotal} entries")
    return index

# Similarity Search Function
def find_similar(input_question, database, index, embedding_model_name, top_k=3):
    # Generate query embedding
    query_embedding = generate_embeddings(input_question, embedding_model_name)
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
