from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from hugchat import hugchat
from hugchat.login import Login
import re
import os
import random
import json
import time
import asyncio
import bson
from datetime import datetime


print("Loading Text Files...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
fulldb = None
maindb = None

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

if not os.path.isfile("creds.txt"):
    print('NO CREDS FILE!')
    exit()
loginlist = loadcreds("creds.txt")

lastcnt = 0
allfiles = []
# print(dir(FAISS))
# dbname = input("DB name:") or "tests"
foldername = input("Folder source name:") or "x-test"
# logincred = input("Login(1-ex3y7jr0r,2-synthfake1,3-synthfake2):") or "1"

# foldername ="1-an_index_of_pupils_and_proteges"
ixnum, tropecat = foldername.split('1-')
tropecat=tropecat.replace("_"," ")

lastconvo={}

if not os.path.isdir("raws"):
    os.mkdir("raws")

if not os.path.isdir("indexes"):
    os.mkdir("indexes")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("cookies_snapshot"):
    os.mkdir("cookies_snapshot")

if not os.path.isdir("done"):
    os.mkdir("done")

ps = list(Path("raws/%s/"%foldername).glob("**/*.txt"))


if os.path.isfile("bson_processed.txt"):
    with open("bson_processed.txt", "r") as pd:
        allfilestr = pd.read()
        allfiles = allfilestr.split('\n')

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

async def hugchat_qa(catname, tropename, querytext,loginname,logincode,lastconvo=lastconvo):
    print(f"Processing {catname}->{tropename}...", end="")
    cookie_path_dir = "./cookies_snapshot"
    newlogin=False
    try:
        sign = Login(loginname,None)
        cookies = sign.loadCookiesFromDir(cookie_path_dir) # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.
    except:
        sign = Login(loginname, logincode)
        cookies = sign.login()
        sign.saveCookiesToDir(cookie_path_dir)
        newlogin=True

    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    
    if newlogin:
        id = chatbot.new_conversation()
        chatbot.change_conversation(id)
        chatbot.switch_llm(4)
        lastconvo[loginname]=id
    else:
        if loginname in lastconvo:
            lastconvoid=lastconvo[loginname]
            chatbot.change_conversation(lastconvoid)

    newdb = None
    cat_index = "indexes/%s.index" % catname
    if os.path.isfile(cat_index):
        newdb = load_vdb(cat_index, embeddings)

    filecontents = []
    metadatas = []
    maindb = None

    try:
        query1 = f'Describe what the trope "{tropename}" is. Discuss what it is about and what it is not about (for disambiguation). Discuss why you should or should not use it, and when to use and when not to use it.'
        query1_ans = chatbot.query(query1 + querytext)
        filecontents.append(query1_ans['text'])
        metadatas.append({'src': cat_index})

        query2 = f'List down the elements needed for something to be classified as this trope, then provide sample templates that can be followed by a writer to use this trope.'
        query2_ans = chatbot.query(query2 + querytext)
        filecontents.append(query2_ans['text'])
        metadatas.append({'src': cat_index})
        
        ndb = FAISS.from_texts(filecontents, embeddings, metadatas=metadatas)
        
        if newdb is None:
            newdb = ndb
        else:
            newdb.merge_from(ndb)

        save_vdb(cat_index,newdb)
        
        tagquery = f"Generate a comma separated list of most commonly used generic instructional phrases that specifically refers to, or suggests the usage of the trope \"{tropename}\" as mentioned above. please write the list only."
        tagquery_ans = chatbot.query(tagquery)
        keyword_string = "%s" % tagquery_ans['text']
        tag_keywords =  [keyword_string]


        keysdb = FAISS.from_texts(tag_keywords, embeddings, metadatas=[{'src':cat_index}])
        maindb_index = "indexes/main.index"

        if os.path.isfile(maindb_index):
            maindb = load_vdb(maindb_index, embeddings)
            print('maindb loaded')

        if maindb is None:
            maindb = keysdb
        else:
            maindb.merge_from(keysdb)

        save_vdb(maindb_index, maindb)

        headertext = f"Key words:\n {keyword_string} \n---\n"
        filecontents.insert(0, headertext)

    except Exception as e:
        print(e)
    
    with open("logs/%s_%s.log"%(catname,tropename), "a+",encoding='utf8') as f:
        resdata = "\n".join(filecontents)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        resdata = f"Updated: {dt_string} \n --- \n" + resdata + "\n --- \n"
        f.write(resdata)

    print("done!")

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

# ---------------------------------------------------test


for ix, p in enumerate(ps):
    strdata = None
    processed = []

    processed_file="done/processed_%s.txt"%tropecat
    if os.path.isfile(processed_file):
        with open(processed_file, "r",encoding="utf8",errors="ignore") as pd:
            allfilestr = pd.read()
            allfiles = allfilestr.split('\n')

    if p.name in allfiles:
        continue

    with open(p, "r",encoding='utf8') as f:
        strdata = f.read()

    strdata = strdata.replace(" ","")
    strdata = strdata.replace("---","\n")
    strpars = strdata.replace("\n\n\n","\n\n")
    tropename = str(p.name).replace(".txt","")
    querytext = f"About the trope {tropename}:\n" + strpars

    loginuse = (ix%3)+1
    lname = loginlist[str(loginuse)]["u"]
    lcode = loginlist[str(loginuse)]["p"]
    print('next hf: %s' % lname)
    
    try:
        asyncio.run(hugchat_qa(tropecat, tropename, querytext, lname, lcode))
        allfiles.append(p.name)
    except Exception as e:
        print(e)
        print('Error with %s!' % p.name)
        exit()

    with open(processed_file, "a+") as pd:
        pd.write(p.name+"\n")

    time.sleep(10)
# ----------------------------------------------------------------------

def filterbytrope(query, cutoffscore=1.5):
    maindb_index = "indexes/main.index"
    maindb = load_vdb(maindb_index, embeddings)

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
        index_res.append(index_src)
        print("%s: %.4f"% (index_src,score))
        # print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
    return index_res

def hflookup(indexpaths,query):
    results = []
    for indexpath in indexpaths:
        dblookup = FAISS.load_local(indexpath,embeddings)
        results_with_scores = dblookup.similarity_search_with_score(query)
        for item in results_with_scores:
            doc, score = item
            index_src=doc.metadata['src']
            result_text=doc.page_content
            results.append(result_text)
            print("Added info: %s\n"% (result_text[0:64]))
            # print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    allresults="".join(results)
    prompt = query + "\n\n You can use the information below as reference:\n" + allresults
    hfanswer = asyncio.run(hugchat_parsequery(prompt, lname, lcode))
    return hfanswer

# querytext="I'd like to write a story about a villain with beastly instincts. Can you create a template for me?"
# dbindexes = filterbytrope(querytext)
# hfresults = hflookup(dbindexes,querytext)

# print(hfresults)

exit()
