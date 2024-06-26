from dotenv import load_dotenv
load_dotenv()

import os
import re
import csv
import json
import time
import numpy as np
import pandas as pd
from pprint import pprint

from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# env setting
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.environ.get('UPSTAGE_API_KEY')
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'matching_model_rag_test' # 프로젝트명 수정
LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT')

print(f'> LangSmith Project: {LANGCHAIN_PROJECT}')


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


# Document 구성
# text
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

# statis
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

# customs
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


# Text Splitter
# statis
splitter = RecursiveCharacterTextSplitter(
    separators=['\r\n', '. ', ', ', ' ', ''],
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
)
statis_splits = splitter.split_documents(statis_documents)

# customs
splitter = RecursiveCharacterTextSplitter(
    separators=['\r\n', '. ', ', ', ' ', ''],
    chunk_size=70,
    chunk_overlap=0,
    length_function=len,
)
customs_splits = splitter.split_documents(customs_documents)


# Vector Store
# Embedding
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model='text-embedding-3-small'
)

# statis
name = 'statis'
folder_path = f'./vectorstore/rag_test/{name}'
if not os.path.exists(folder_path):
    print(f'> "{folder_path}" 생성 중')
    statis_vectorstore = FAISS.from_documents(
        documents=statis_splits,
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

# customs
name = 'customs'
folder_path = f'./vectorstore/rag_test/{name}'
if not os.path.exists(folder_path):
    print(f'> "{folder_path}" 생성 중')
    customs_vectorstore = FAISS.from_documents(
        documents=customs_splits,
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


# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
# prompt = hub.pull("rlm/rag-prompt")
templete="""You are an assistant who explanation the question in Korean. Use the following pieces of retrieved context to answer the question. The retrieved context is useful information for explanation. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:"""

prompt = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'], 
                template=templete
            )
        )
    ]
)


# LLM과 검색엔진을 활용한 RAG 구현
retriever = customs_vectorstore.as_retriever(search_kwargs={"k": 5})
chat = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)

def format_docs(docs):
    global references
    references = docs
    return '\n\n'.join(doc.page_content for doc in docs)

def answer_question(messages):
    global references
    response = {"topk": "", "answer": "", "references": ""}

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | chat
        | StrOutputParser()
    )

    ans = rag_chain.invoke(messages)

    topk = [reference.metadata['HS_CODE'] for reference in references]
    ref = [
        {
            'page_content': reference.page_content,
            'metadata': reference.metadata
        } for reference in references
    ]

    response["topk"] = topk
    response["answer"] = ans
    response["references"] = ref

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as eval_lines, open(output_filename, 'w') as output_lines:
        for eval_line in eval_lines:
            j = json.loads(eval_line)
            print(f'CODE: {j["CODE"]}\nQuestion: {j["DSC"]}')
            response = answer_question(j["DSC"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"CODE": j["CODE"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            output_lines.write(f'{json.dumps(output, ensure_ascii=False)}\n')

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag('../data/jsonl_prepro_text.jsonl', '../submit/rag_test.csv')

# LangSmith 저장 시간 확보
time.sleep(60)