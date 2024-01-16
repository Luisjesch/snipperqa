from pathlib import Path
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from hugchat import hugchat
from hugchat.login import Login
import re
import os
import random
import json
import time
import asyncio
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
foldername = input("Folder source name:") or "x-test"
# logincred = input("Login(1-ex3y7jr0r,2-synthfake1,3-synthfake2):") or "1"
ixnum, tropecat = foldername.split('1-')
tropecat=tropecat.replace("_"," ")

lastconvo={}
if not os.path.isdir("raws"):
    os.mkdir("raws")

if not os.path.isdir("hfinfer"):
    os.mkdir("hfinfer")

if not os.path.isdir("hfinfer/data"):
    os.mkdir("hfinfer/data")

if not os.path.isdir("hfinfer/keys"):
    os.mkdir("hfinfer/keys")

if not os.path.isdir("indexes"):
    os.mkdir("indexes")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("cookies_snapshot"):
    os.mkdir("cookies_snapshot")

if not os.path.isdir("done"):
    os.mkdir("done")

ps = list(Path("raws/%s/"%foldername).glob("**/*.txt"))

def load_text(indexpath):
    if os.path.isfile(indexpath):
        with open(indexpath, "r") as pd:
            filetext = pd.read()
            filepars = dbjson.split("\n****\n") 
            return filepars

def save_text(indexpath, pars):
    filetext = "\n****\n".join(pars)
    with open(indexpath, "w") as pd:
        pd.write(filetext)

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

    textcontent = []

    hfdir = "hfinfer/data/%s"%catname
    keysdir = "hfinfer/keys/%s"%catname

    if not os.path.isdir(hfdir):
        os.mkdir(hfdir)

    if not os.path.isdir(keysdir):
        os.mkdir(keysdir)

    file_index = "%s/%s.txt" % (hfdir,tropename)
    keys_index = "%s/%s.txt" % (keysdir,tropename)

    try:
        ins_txt = "\nUse the previous outputs and the information below as reference:\n"
        query1 = f'Briefly describe what the trope "{tropename}" is in detail. Discuss what it is about and what it is not about (for disambiguation).'
        query1_ans = chatbot.query(query1 + ins_txt + querytext)
        textcontent.append(query1_ans['text'])

        query2 = f'Briefly discuss why you should or should not use the trope,\"{tropename}\".'
        query2_ans = chatbot.query(query2 + ins_txt + querytext)
        textcontent.append(query2_ans['text'])

        query3 = f'Briefly discuss when to use and when not to use the trope,\"{tropename}\".'
        query3_ans = chatbot.query(query3 + ins_txt + querytext)
        textcontent.append(query3_ans['text'])

        query3 = f'Briefly discuss how to effectively and properly use the trope,\"{tropename}\".'
        query3_ans = chatbot.query(query3 + ins_txt + querytext)
        textcontent.append(query3_ans['text'])

        query2 = f'List down the elements needed for something to be classified as the trope,\"{tropename}\".'
        query2_ans = chatbot.query(query2 + ins_txt + querytext)
        textcontent.append(query2_ans['text'])

        query3 = f'Give me different unique templates which can be used as a guide for writers in using the trope,\"{tropename}\".'
        query3_ans = chatbot.query(query3 + ins_txt + querytext)
        textcontent.append(query3_ans['text'])

        save_text(file_index, textcontent)
        
        tagquery = f"Generate a list of most commonly used keywords or instructional phrases that specifically refers to, or suggests the usage of the trope \"{tropename}\" as mentioned above. please write the list only."
        tagquery_ans = chatbot.query(tagquery  + ins_txt + querytext)
        keyword_string = "%s" % tagquery_ans['text']
        keystext = f"{keyword_string}"

        save_text(keys_index, [keystext])

    except Exception as e:
        print(e)
    
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
        with open(processed_file, "a+") as pd:
            pd.write(p.name+"\n")
    except Exception as e:
        print(e)
        print('Error with %s!' % p.name)
        exit()

    if ix == 10:
        exit()

# ----------------------------------------------------------------------

exit()
