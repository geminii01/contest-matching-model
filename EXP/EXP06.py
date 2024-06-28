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
text = pd.read_excel('../data/비식별된 해외기업별 영문 텍스트데이터.xlsx')
statis = pd.read_excel('../data/통계청 국제표준산업분류 HSCODE 6단위 매핑.xlsx')
customs = pd.read_excel('../data/관세청_HS부호_240101.xlsx')

text_copy = text.copy()
statis_copy = statis.copy()
customs_copy = customs.copy()

print('> 데이터 로드 완료')


# 데이터 전처리
def zero_input(num, x):
    if pd.isna(x):
        return np.nan
    else:
        cnt = num - len(x)
        return '0' * cnt + x
    
def re_sub(x):
    if pd.isna(x):
        return np.nan
    else:
        return re.sub(r'^\((.*?)\)$', r'\1', x)

text_copy['ID'] = text_copy['ID'].astype(str)
text_copy['CODE'] = text_copy['CODE'].astype(str)
text_copy['CODE'] = text_copy['CODE'].apply(lambda x: zero_input(4, x))

statis_copy.columns = [
    'ISIC4_CODE', # ISIC4_국제표준산업분류
    'ISIC4_NAME', # ISIC4_분류명
    'KSIC10_CODE', # KSIC10_한국표준산업분류
    'KSIC10_NAME', # KSIC10_분류명
    'HS2017_CODE', # HS2017_관세통계통합품목분류
    'HS2017_NAME' # HS2017_분류명
]

statis_copy['ISIC4_CODE'] = statis_copy['ISIC4_CODE'].astype(str)
statis_copy['ISIC4_CODE'] = statis_copy['ISIC4_CODE'].replace('nan', np.nan)
statis_copy['ISIC4_CODE'] = statis_copy['ISIC4_CODE'].str.replace('.0', '', regex=False)
statis_copy['ISIC4_CODE'] = statis_copy['ISIC4_CODE'].apply(lambda x: zero_input(4, x))

statis_copy['HS2017_CODE'] = statis_copy['HS2017_CODE'].astype(str)
statis_copy['HS2017_CODE'] = statis_copy['HS2017_CODE'].replace('nan', np.nan)
statis_copy['HS2017_CODE'] = statis_copy['HS2017_CODE'].str.replace('.0', '', regex=False)
statis_copy['HS2017_CODE'] = statis_copy['HS2017_CODE'].apply(lambda x: zero_input(6, x))

customs_copy.columns = [
    'HS_CODE', # HS부호
    'KOR_NAME', # 한글품목명
    'ENG_NAME', # 영문품목명
    'INT_CODE', # 성질통합분류코드
    'INT_NAME' # 성질통합분류명
]

customs_copy['HS_CODE'] = customs_copy['HS_CODE'].astype(str)
customs_copy['HS_CODE'] = customs_copy['HS_CODE'].apply(lambda x: zero_input(10, x))

customs_copy['INT_CODE'] = customs_copy['INT_CODE'].astype(str)
customs_copy['INT_CODE'] = customs_copy['INT_CODE'].replace('nan', np.nan)
customs_copy['INT_CODE'] = customs_copy['INT_CODE'].str.replace('.0', '', regex=False)

customs_copy['INT_NAME'] = customs_copy['INT_NAME'].apply(lambda x: re_sub(x))

text_copy = text_copy.fillna(' ')
statis_copy = statis_copy.fillna(' ')
customs_copy = customs_copy.fillna(' ')

print('> 데이터 전처리 완료')
print('> 데이터 결측치 확인')
print('-----' * 5)
print(text_copy.isnull().sum())
print(statis_copy.isnull().sum())
print(customs_copy.isnull().sum())
print('-----' * 5)


# 데이터 저장 및 로드
text_copy.to_csv('../data/prepro_text.csv', index=False, encoding='utf-8')
statis_copy.to_csv('../data/prepro_statis.csv', index=False, encoding='utf-8')
customs_copy.to_csv('../data/prepro_customs.csv', index=False, encoding='utf-8')

text_prepro = pd.read_csv('../data/prepro_text.csv', dtype=str)
statis_prepro = pd.read_csv('../data/prepro_statis.csv', dtype=str)
customs_prepro = pd.read_csv('../data/prepro_customs.csv', dtype=str)


# csv to jsonl
def csv_to_jsonl(csv_file_path, jsonl_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in csv_reader:
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + '\n')

csv_to_jsonl('../data/prepro_text.csv', '../data/jsonl_prepro_text.jsonl')
csv_to_jsonl('../data/prepro_statis.csv', '../data/jsonl_prepro_statis.jsonl')
csv_to_jsonl('../data/prepro_customs.csv', '../data/jsonl_prepro_customs.jsonl')
print('> csv to jsonl 완료')


# Document 구성 - text
file_path = '../data/jsonl_prepro_text.jsonl'
temp = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        temp.append(json.loads(line.strip()))

seq_num = 1
text_documents = []
for data in temp:
    doc = Document(
        page_content=data['DSC'], 
        metadata={
            'ID': data['ID'],
            'CODE': data['CODE'],
            'source': '/root/contest-matching-model/data/jsonl_prepro_text.jsonl',
            'seq_num': seq_num,
        }
    )
    text_documents.append(doc)
    seq_num += 1


# Document 구성 - statis
file_path = '../data/jsonl_prepro_statis.jsonl'
temp = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        temp.append(json.loads(line.strip()))

seq_num = 1
statis_documents = []
for data in temp:
    doc = Document(
        page_content=f"{data['ISIC4_NAME']}\r\n{data['KSIC10_NAME']}\r\n{data['HS2017_NAME']}", # ISIC4, KSIC10, HS2017 순으로 작성됨
        metadata={
            'ISIC4_CODE': data['ISIC4_CODE'],
            'KSIC10_CODE': data['KSIC10_CODE'],
            'HS2017_CODE': data['HS2017_CODE'],
            'source': '/root/contest-matching-model/data/jsonl_prepro_statis.jsonl',
            'seq_num': seq_num,
        }
    )
    statis_documents.append(doc)
    seq_num += 1


# Document 구성 - customs
file_path = '../data/jsonl_prepro_customs.jsonl'
temp = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        temp.append(json.loads(line.strip()))

seq_num = 1
customs_documents = []
for data in temp:
    doc = Document(
        page_content=f"{data['KOR_NAME']}\r\n{data['ENG_NAME']}\r\n{data['INT_NAME']}", # 한글품목명, 영어품목명, 성질 통합 분류명 순으로 작성됨
        metadata={
            'HS_CODE': data['HS_CODE'],
            'INT_CODE': data['INT_CODE'],
            'source': '/root/contest-matching-model/data/jsonl_prepro_customs.jsonl',
            'seq_num': seq_num,
        }
    )
    customs_documents.append(doc)
    seq_num += 1
print('> Document 구성 완료')


# OpenAI Summarization + Translation

# 프롬프트 준비
def summary_prompt(text):
    prompt = (
        "Please summarize the following English text, ensuring that the summary captures the main points and key information from the original text:\n\n"
        "{text}\n\n"
        "Then translate the summary into Korean.\n\n"
        "Desired output format:\n"
        "Summary: <English summary>\n"
        "Korean: <Korean translation>"
    )
    return prompt.format(text=text)

# 요약 및 번역 결과 반환
def get_summary(text):
    prompt = summary_prompt(text=text)
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        seed=1
    )
    return completion.choices[0].message.content

# KEY setting
client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_KEY
)

# jsonl 형식으로 데이터 저장
# 데이터는 DSC 기준 drop_duplicates 적용한 것으로 준비
text_dup = text_prepro.drop_duplicates(subset='DSC').reset_index(drop=True)
file_path = '../data/jsonl_summary_text_EXP06.jsonl'
with open(file_path, 'a', encoding='utf-8') as file:
    for idx in tqdm(iterable=range(text_dup.shape[0]), desc='summarization and translation'):
        text = text_dup.loc[idx, 'DSC']
        response = get_summary(text=text)

        output = {
            "ID": text_dup.loc[idx, 'ID'], 
            "CODE": text_dup.loc[idx, 'CODE'], 
            "DSC": response
        }
        file.write(f'{json.dumps(output, ensure_ascii=False)}\n')

# 요약과 번역을 각각의 컬럼으로 분리
data = []
text_dup_sum_enko = text_dup.copy()
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
for i in tqdm(iterable=range(text_sum_enko.shape[0]), desc='영문 텍스트 데이터 재구성'):
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
print('> Document 재구성 완료')

# Text Splitter 생략


# Embedding
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model='text-embedding-3-large'
)


# Vector Store 생성 - statis
name = 'statis'
folder_path = f'./vectorstore/EXP06/{name}'
if not os.path.exists(folder_path):
    print(f'> "{folder_path}" 생성 중')
    statis_vectorstore = FAISS.from_documents(
        documents=statis_documents,
        embedding=embeddings,
    )
    statis_vectorstore.save_local(folder_path=folder_path)
    print(f'> "{folder_path}" 생성 및 로컬 저장 완료')
else:
    statis_vectorstore = FAISS.load_local(
        folder_path=folder_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    print(f'> "{folder_path}" 로컬에서 불러옴')


# Vector Store 생성 - customs
name = 'customs'
folder_path = f'./vectorstore/EXP06/{name}'
if not os.path.exists(folder_path):
    print(f'> "{folder_path}" 생성 중')
    customs_vectorstore = FAISS.from_documents(
        documents=customs_documents,
        embedding=embeddings,
    )
    customs_vectorstore.save_local(folder_path=folder_path)
    print(f'> "{folder_path}" 생성 및 로컬 저장 완료')
else:
    customs_vectorstore = FAISS.load_local(
        folder_path=folder_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    print(f'> "{folder_path}" 로컬에서 불러옴')


# Information Retrieve

# vectorstore as retriever
statis_retriever = statis_vectorstore.as_retriever(
    search_type='mmr', 
    search_kwargs={
        'k': 3, 
        'fetch_k': 50, 
        'lambda_mult': 1.0,
    }
)
customs_retriever = customs_vectorstore.as_retriever(
    search_type='mmr', 
    search_kwargs={
        'k': 5, 
        'fetch_k': 50, 
        'lambda_mult': 1.0,
    }
)

# Create directory
statis_file_path = '../submit/EXP06/statis.jsonl'
os.makedirs(os.path.dirname(statis_file_path), exist_ok=True)
customs_file_path = '../submit/EXP06/customs.jsonl'
os.makedirs(os.path.dirname(customs_file_path), exist_ok=True)

# Information Retrieval
for text_document in tqdm(iterable=text_documents, desc='Information Retrieval'):
    # query
    query = text_document.page_content

    # statis retriever
    statis_results = statis_retriever.invoke(query)

    # Save References and Create new query
    idx = 0
    statis_query = ''
    with open(statis_file_path, 'a', encoding='utf-8') as statis_ref:
        statis_references = {
            "ISIC4_CODE": "", 
            "KSIC10_CODE": "", 
            "HS2017_CODE": "", 
            "result": ""
        }        
        isic4 = [reference.metadata['ISIC4_CODE'] for reference in statis_results]
        ksic10 = [reference.metadata['KSIC10_CODE'] for reference in statis_results]
        hs2017 = [reference.metadata['HS2017_CODE'] for reference in statis_results]
        result = [reference.page_content for reference in statis_results]
        statis_references['ISIC4_CODE'] = isic4
        statis_references['KSIC10_CODE'] = ksic10
        statis_references['HS2017_CODE'] = hs2017
        statis_references['result'] = result

        statis_ref.write(f'{json.dumps(statis_references, ensure_ascii=False)}\n')

        for res in statis_results:
            statis_query += f'\n{res.page_content}'
        
        # customs retriever
        customs_results = customs_retriever.invoke(statis_query)

        # Save References
        with open(customs_file_path, 'a', encoding='utf-8') as customs_ref:
            customs_references = {
                "HS_CODE": "", 
                "INT_CODE": "", 
                "result": ""
            }        
            hscode = [reference.metadata['HS_CODE'] for reference in customs_results]
            intcode = [reference.metadata['INT_CODE'] for reference in customs_results]
            result = [reference.page_content for reference in customs_results]
            customs_references['HS_CODE'] = hscode
            customs_references['INT_CODE'] = intcode
            customs_references['result'] = result

            customs_ref.write(f'{json.dumps(customs_references, ensure_ascii=False)}\n')