# 실험 설명

## EXP01

- Retriever 검색 성능 확인

    - Text Splitter: `RecursiveCharacterTextSplitter`

    - Vector Store: `FAISS`

    - Search Type: `Similarity`

    - Embedding Model: `solar-embedding-1-large`

## EXP02

- Embedding Model 변경

    - `solar-embedding-1-large` → `text-embedding-3-large`

## EXP03

- Embedding Model 변경

    - `text-embedding-3-large` → `text-embedding-ada-002`

## EXP04

- Ensemble Retriever

    - Sparse Retriever로 `BM25` 사용

## EXP05

- Summarization과 Translation으로 영문 텍스트 데이터 재구성

    - LLM Model: `gpt-3.5-turbo-0125`

## EXP06

- Search Type 변경

    - `Similarity` → `MMR`

## EXP07 (진행 중)

- 주최 측 제공 데이터 대신, 팀원이 크롤링으로 수집한 HS 해설 데이터 사용

    - 주최 측 데이터 결측치, 이상치 해결 가능

- LangGraph 활용

    - 답변 생성까지 필요없으나, 영문 텍스트와 Retriever의 관련성을 유연하게 파악 가능