#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hashlib
import random
import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from transformers import AutoTokenizer
# from huggingface_hub import login
import torch
import time
from Model_and_Chroma import tokenizer,model,collection_for_documents
import tiktoken

def read_file(docs):
    documents = []
    for file in docs:
        temp_file = f"./{file.name}"
        with open(temp_file, "wb") as f:
            f.write(file.getvalue())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file)
            documents.extend(loader.load())
        elif file.name.endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(temp_file)
            documents.extend(loader.load())
        elif file.name.endswith(".txt"):
            loader = TextLoader(temp_file)
            documents.extend(loader.load())
    for doc in documents:
        doc.page_content = doc.page_content.replace('\n', '')
    # print("Documents:",documents)
    return documents

def chunking(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    return text_splitter.split_documents(document)

def storing(documents, collection_for_documents):
    id, data, metadata = [], [], []
    for i, doc in enumerate(documents):
        id.append(create_document_id(doc.metadata['source'],doc.page_content))
        data.append(doc.page_content)
        metadata.append(doc.metadata)
    # print("Ids:",ids)
    collection_for_documents.upsert(
        documents=data,
        metadatas=metadata,
        ids=id
    )
    # print("In storage check if has stored")
    # print(collection_for_documents.get(ids=id))
    
def count_tokens(context):
    data=''
    for i in context:
        data+=i
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(data))
    return num_tokens

def create_document_id(file_name, text):
    # Create a unique hash based on file name and its text
    hash_object = hashlib.sha256(f"{file_name}{text}".encode())
    document_id = hash_object.hexdigest()
    return document_id


def generate_answer(question, collection_for_documents, tokenizer, model):
    context = collection_for_documents.query(
        query_texts=[question],
        n_results=4,
    )['documents'][0]
    # print("Context:",context)
    num_tokens = count_tokens(context)
    print("Tokens:",num_tokens)
#     print(context)
#     system_message = f"""Answer the question based on the context.Respond "Unsure about answer" if not sure about the answer.
#             CONTEXT:{context}
#             QUESTION: {question}

#             """
    system_message=f"""Explain the answer of the question from the context:{context} Question:{question}
       Respond "Unsure about answer" if not sure about the answer.
            """
    min=int(0.1*num_tokens)
    max=int(0.3*num_tokens)
    responses=[]
    for i in range(5):
        inputs = tokenizer(system_message, return_tensors="pt")
        outputs = model.generate(**inputs,min_new_tokens=min,max_new_tokens=max,do_sample=True,temperature=random.uniform(0.5, 1.5))
        output_text=tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.append(output_text[0])
    print(responses)    
    
    system_message_final=f"""Generate a final answer from the given list of responses:{responses}
            """
    input = tokenizer(system_message_final, return_tensors="pt")
    output = model.generate(**input,min_new_tokens=min,max_new_tokens=max,do_sample=True,temperature=0.1)
    final_output=tokenizer.batch_decode(output, skip_special_tokens=True)
#     print("output: ")
#     print(output_text)
    return final_output
        
           
# Set the title for the Streamlit app
st.title("Upload a File to Get Started - 🤖🔗")
  
with st.sidebar:
    uploaded_files = st.file_uploader("Upload File", type=["pdf", "doc", "txt", "docx"], accept_multiple_files=True)
    if uploaded_files:
        upload_start=time.time()
        with st.spinner('File(s) is/are being processed...'):
        #     print("read_file")
            document = read_file(uploaded_files)
            documents = chunking(document)
        #     print("store")
            storing(documents, collection_for_documents)
        #     # print(documents)
            # print("HashValues")
            
            # for doc in document:
            #     # print(type(doc.page_content))
            #     document_id = create_document_id(doc.metadata['source'], doc.page_content)
            #     print(document_id," ",collection_for_documents.get(ids=document_id))
            #     if collection_for_documents.get(ids=document_id)['ids']==[]:
            #     #     print("chunk_file")
            #         documents = chunking(document)
            #         # print("store")
            #         storing(documents, collection_for_documents)
            #         # print(documents)
        st.success('Done!')
        print("Files upload time",time.time()-upload_start)


# Initialize Streamlit chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
start=time.time()
print("Question")
if prompt := st.chat_input("Ask your questions from File",disabled=not uploaded_files):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner('Generating answer...'):
        with st.chat_message("user"):
            st.markdown(prompt)
        
            result=generate_answer(prompt,collection_for_documents,tokenizer,model)
            # print(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result[0]
            message_placeholder.markdown(full_response + "|")
    message_placeholder.markdown(full_response)
    # print(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    print("Question time:",time.time()-start)

# In[ ]:




