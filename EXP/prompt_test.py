from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import pandas as pd
from openai import OpenAI


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_ORG_KEY = os.environ.get('OPENAI_ORG_KEY')


text_prepro = pd.read_csv('../data/prepro_text.csv', dtype=str)


def summary_prompt(text):
    prompt = (
        "제공된 Context는 영어로 되어있으며, 해외 기업이 취급하고 있는 품목 및 산업에 대한 설명입니다."
        " 제공된 Context에서 취급하고 있는 품목 및 산업에 대한 주요 정보를 한국어로 번역한 후 최대 100자 이내로 요약해주세요."
        "\n답변 예시:\n"
        "이 해외 기업은 ... 산업으로, 품목은 ...을(를) 취급하고 있습니다.\n"
        "이 해외 기업은 ... 산업으로, ...에 관한 정보를 제공합니다.\n"
        f"\nContext: {text}\n"
        "Answer:"
    )
    return prompt.format(text=text)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_KEY
)

def get_summary(text):
    prompt = summary_prompt(text=text)
    response = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        seed=1
    )
    return response.choices[0].message.content

text_dup = text_prepro.drop_duplicates(subset='DSC').reset_index(drop=True)
file_path = '../data/jsonl_summary_text_EXP08.jsonl'
with open(file_path, 'w', encoding='utf-8') as file:
    for idx in range(text_dup.shape[0]):
        text = text_dup.loc[idx, 'DSC']
        response = get_summary(text=text)

        output = {
            "ID": text_dup.loc[idx, 'ID'], 
            "CODE": text_dup.loc[idx, 'CODE'], 
            "DSC": response
        }
        file.write(f'{json.dumps(output, ensure_ascii=False)}\n')

        print(f'[{idx}]', '-----' * 15)
        print(response)

        time.sleep(0.5)