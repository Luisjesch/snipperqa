from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

import os
import streamlit as st
from streamlit_chat import message
import random
import json
import time
import asyncio
from datetime import datetime

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
fulldb = None
maindb = None

lastcnt = 0
allfiles = []


def load_vdb(indexpath, embeddings):
    if os.path.isfile(indexpath):
        with open(indexpath, "rb") as pd:
            vdb_pkl = pd.read()
            vdb = FAISS.deserialize_from_bytes(
                embeddings=embeddings, serialized=vdb_pkl
                )
            return vdb

def save_vdb(indexpath, vdb):
    vdb_pkl = vdb.serialize_to_bytes()
    with open(indexpath, "wb") as pd:
        pd.write(vdb_pkl)


def filterbytrope(query, cutoffscore=1.5):
    maindb_index = "indexes/main.index"
    maindb = load_vdb(maindb_index,embeddings)

    # embedding_vector = embeddings.embed_query("give me an example of the pupil master trope?")
    # results_with_scores = maindb.similarity_search_by_vector(embedding_vector,k=10)
    results_with_scores = maindb.similarity_search_with_score(query)
    index_res = []
    for item in results_with_scores:
        # doc, metad, score = item
        doc, score = item
        # if score > cutoffscore:
        #     continue
        index_src=doc.metadata['src']
        if not index_src in index_res:
            index_res.append(index_src)
            print("%s: %.4f"% (index_src,score))
        # print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
    return index_res

test_search = input("Search:") 
dbindexes = filterbytrope(test_search)
