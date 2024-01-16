from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from hugchat import hugchat
from hugchat.login import Login

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

def loadcreds(creds_filepath):    
    if os.path.isfile(creds_filepath):
        cred_dict = {}
        with open(creds_filepath, "r") as pd:
            creddata = pd.read()
            creds = creddata.split('\n')
            for cix, cred in enumerate(creds):
                cred_id = str(cix + 1)
                cred_login, cred_pass = cred.split('|')
                cred_dict[cred_id]={"u":cred_login,"p":cred_pass}
        return cred_dict
        
loginlist = loadcreds("creds.txt")


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

async def hugchat_parsequery(querytext,loginname,logincode):

    cookie_path_dir = "./cookies_snapshot"
    try:
        sign = Login(loginname,None)
        cookies = sign.loadCookiesFromDir(cookie_path_dir) # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.
    except:
        sign = Login(loginname, logincode)
        cookies = sign.login()
        sign.saveCookiesToDir(cookie_path_dir)
        id = chatbot.new_conversation()
        chatbot.change_conversation(id)
        chatbot.switch_llm(4)
    
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

    try:
        query_ans = chatbot.query(querytext)
        return query_ans
    except Exception as e:
        print(e)
    return None

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

async def hflookup(indexpaths,query,lname,lcode):
    results = []
    for indexpath in indexpaths:
        dblookup = load_vdb(indexpath,embeddings)
        results_with_scores = dblookup.similarity_search_with_score(query)
        for item in results_with_scores:
            doc, score = item
            index_src=doc.metadata['src']
            result_text=doc.page_content
            results.append(result_text)
            print("info(%.4f): %s\n"% (score,result_text[0:64]))
            # print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    allresults="".join(results)
    prompt = query + "\n\n You can use the information below as reference:\n" + allresults
    hfanswer = await hugchat_parsequery(prompt, lname, lcode)
    return hfanswer['text']

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Docs Chat", page_icon=":robot:")
st.header("Docs Chat")

if "generated" not in st.session_state:
    st.session_state["generated"] = []
    st.session_state["generated"].append({"role": 'assistant', "text": 'Hello! What can I help you with?',"front": "Greetings!"})

lastitem = len(st.session_state.generated) - 1
for i in range(0, len(st.session_state.generated)):
    message = st.session_state.generated[i]
    role = message['role']
    text = message['text']
    front = message['front']
    islastitem = i == lastitem
    with st.expander(front, expanded=islastitem):
        with st.chat_message(role):
            st.markdown(text)

user_input = st.chat_input(placeholder="What did the fox say?")

if user_input:
    usermsg = {"role": 'user', "text": str(user_input), "front":str(user_input)[0:80]}
    st.session_state.generated.append(usermsg)
    timenow = time.time()
    res_pars = []

    dbindexes = filterbytrope(user_input)
    loginuse = (len(st.session_state.generated)%3)+1
    lname = loginlist[str(loginuse)]["u"]
    lcode = loginlist[str(loginuse)]["p"]
    hfresults = asyncio.run(hflookup(dbindexes,user_input,lname,lcode))

    if hfresults:
        assistant_message = {"role": 'assistant', "text": hfresults, "front": hfresults[0:64]}
        st.session_state.generated.append(assistant_message)
        st.balloons()
        st.rerun()