#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[ ]:


# get_ipython().system('pip install streamlit')


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoModel
import torch
import chromadb
import time
from chromadb.config import Settings
from chromadb.utils import embedding_functions
# from langchain.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer


start_time=time.time()
# print("create_db")
client = chromadb.PersistentClient(path="./Database")
embeddings_model = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="./EmbeddingModel"    # Provide the pre-trained model's path
    # model_kwargs=model_kwargs, # Pass the model configuration options
    # encode_kwargs=encode_kwargs # Pass the encoding options
)

collection_for_documents = client.get_or_create_collection(name="Storage", embedding_function=embeddings_model, metadata={"hnsw:space": "cosine"}) # type: ignore
# print("made db")


model = AutoModelForSeq2SeqLM.from_pretrained("./Model/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a")
tokenizer = AutoTokenizer.from_pretrained("./Tokenizer")
print("Model time:",time.time()-start_time)
