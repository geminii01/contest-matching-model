from dotenv import load_dotenv
load_dotenv()

import os
import re
import csv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# env setting
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_ORG_KEY = os.environ.get('OPENAI_ORG_KEY')


# 데이터 로드
text_prepro = pd.read_csv('../data/prepro_text.csv', dtype=str)
hs2 = pd.read_csv('../data/HS_2.csv', dtype=str, encoding='utf-8')
hs4 = pd.read_csv('../data/HS_4.csv', dtype=str, encoding='utf-8')
hs5 = pd.read_csv('../data/HS_5.csv', dtype=str, encoding='utf-8')
hs6 = pd.read_csv('../data/HS_6.csv', dtype=str, encoding='utf-8')
hs8 = pd.read_csv('../data/HS_8.csv', dtype=str, encoding='utf-8')
hs10 = pd.read_csv('../data/HS_10.csv', dtype=str, encoding='utf-8')

print('> Data Load')


# Create HSCODE Documents
def get_unique_value(df_0, idx, col_0, col_1, col_2, num):
    val = df_0[df_0[col_0] == hs10.loc[idx, col_1][:num]][col_2].unique()
    if len(val) == 1:
        return val[0]
    return ''

hs5['HS_5_'] = hs5['HS_4'] + hs5['HS_5']
hs6['HS_6_'] = hs6['HS_4'] + hs6['HS_6']
hs8['HS_8_'] = hs8['HS_4'] + hs8['HS_6'] + hs8['HS_8']
hs10['HS_10_'] = hs10['HS_4'] + hs10['HS_6'] + hs10['HS_10']

seq_num = 1
documents = []
for idx in tqdm(range(hs10.shape[0]), desc='Create HSCODE Documents'):
    a = hs10.loc[idx, 'KOR']
    b = hs10.loc[idx, 'ENG']
    c = get_unique_value(hs2, idx, 'HS_2', 'HS_10_', 'BU', 2)
    d = get_unique_value(hs2, idx, 'HS_2', 'HS_10_', 'RYU', 2)
    e = get_unique_value(hs4, idx, 'HS_4', 'HS_10_', 'KOR', 4)
    f = get_unique_value(hs4, idx, 'HS_4', 'HS_10_', 'ENG', 4)
    g = get_unique_value(hs5, idx, 'HS_5_', 'HS_10_', 'KOR', 5)
    h = get_unique_value(hs5, idx, 'HS_5_', 'HS_10_', 'ENG', 5)
    i = get_unique_value(hs6, idx, 'HS_6_', 'HS_10_', 'KOR', 6)
    j = get_unique_value(hs6, idx, 'HS_6_', 'HS_10_', 'ENG', 6)
    k = get_unique_value(hs8, idx, 'HS_8_', 'HS_10_', 'KOR', 8)
    l = get_unique_value(hs8, idx, 'HS_8_', 'HS_10_', 'ENG', 8)

    doc = Document(
        page_content=f'[품명]\n- {a} ({b})\n\n[부 해설서]\n- {c}\n\n[류 해설서]\n- {d}\n\n[호 해설서]\n- {e} ({f})\n\n[추가 해설서]\n- {g} ({h})\n- {i} ({j})\n- {k} ({l})',
        metadata={
            'HSCODE': hs10.loc[idx, 'HS_10_'],
            'seq_num': seq_num
        }
    )
    documents.append(doc)
    seq_num += 1
print(documents[0].page_content)
print(documents[0].metadata)


# 요약과 번역을 각각의 컬럼으로 분리
text_dup = text_prepro.drop_duplicates(subset='DSC').reset_index(drop=True)
text_dup_sum_enko = text_dup.copy()

data = []
file_path = '../data/jsonl_summary_text_EXP06.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)
cond = df['DSC'].apply(lambda x: x.split('\n\nKorean: '))
for idx in range(len(cond)):
    text_dup_sum_enko.loc[idx, 'DSC_summary'] = cond[idx][0].strip('Summary: ')
    text_dup_sum_enko.loc[idx, 'DSC_enko'] = cond[idx][1]

# drop_duplicates를 적용한 데이터를 활용하였으므로, 각 요약과 번역을 원래의 row 길이에 맞게 설정
text_sum_enko = text_prepro.copy()
for i in tqdm(iterable=range(text_sum_enko.shape[0]), desc='Reconstruction ENG Text'):
    for j in range(text_dup_sum_enko.shape[0]):
        if text_sum_enko.loc[i, 'DSC'] == text_dup_sum_enko.loc[j, 'DSC']:
            text_sum_enko.loc[i, 'DSC_summary'] = text_dup_sum_enko.loc[j, 'DSC_summary']
            text_sum_enko.loc[i, 'DSC_enko'] = text_dup_sum_enko.loc[j, 'DSC_enko']

        
# Document 재구성 - text
seq_num = 1
text_documents = []
for idx in range(text_sum_enko.shape[0]):
    doc = Document(
        page_content=f"{text_sum_enko.loc[idx, 'DSC_summary']}\n{text_sum_enko.loc[idx, 'DSC_enko']}", 
        metadata={
            'ID': text_sum_enko.loc[idx, 'ID'],
            'CODE': text_sum_enko.loc[idx, 'CODE'],
            'source': '/root/contest-matching-model/data/jsonl_summary_text.jsonl',
            'seq_num': seq_num,
        }
    )
    text_documents.append(doc)
    seq_num += 1
print('> Create ENG Text Documents')


# Text Splitter 생략


# Embedding
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model='text-embedding-3-large'
)


# Vector Store 생성
folder_path = f'./vectorstore/EXP07'
if not os.path.exists(folder_path):
    print(f'> "{folder_path}" ...')
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    vectorstore.save_local(folder_path=folder_path)
    print(f'> Create and Save "{folder_path}"')
else:
    vectorstore = FAISS.load_local(
        folder_path=folder_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    print(f'> Load "{folder_path}"')


# vectorstore as retriever
retriever = vectorstore.as_retriever(
    search_type='mmr', 
    search_kwargs={
        'k': 5, 
        'fetch_k': 100,
        'lambda_mult': 0.95
    }
)


# Create directory
file_path = '../submit/EXP07/hscode.jsonl'
os.makedirs(os.path.dirname(file_path), exist_ok=True)


# Information Retrieval
for text_document in tqdm(iterable=text_documents, desc='Information Retrieval'):
    # query
    query = text_document.page_content

    # retriever
    results = retriever.invoke(query)

    # Save References and Create new query
    with open(file_path, 'a', encoding='utf-8') as ref:
        references = {
            "HSCODE": "", 
            "seq_num": "",
            "result": ""
        }
        ref_0 = [reference.metadata['HSCODE'] for reference in results]
        ref_1 = [reference.metadata['seq_num'] for reference in results]
        ref_2 = [reference.page_content for reference in results]
        references['HSCODE'] = ref_0
        references['seq_num'] = ref_1
        references['result'] = ref_2

        ref.write(f'{json.dumps(references, ensure_ascii=False)}\n')