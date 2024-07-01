import os
import pandas as pd


file_path = './data'
os.makedirs(os.path.dirname(file_path), exist_ok=True)


# # 해외 기업 설명 데이터와 통계청 데이터의 ICIS4 기준으로 HS부호 6자리 매칭
text_data = pd.read_excel('./data/비식별된 해외기업별 영문 텍스트데이터.xlsx', dtype=str)
statis_data = pd.read_excel('./data/통계청 국제표준산업분류 HSCODE 6단위 매핑.xlsx', dtype=str)
dup_text_data = text_data.drop_duplicates(subset='CODE').reset_index(drop=True)

temp = []
for i in range(dup_text_data.shape[0]):
    isic4_code = dup_text_data.loc[i, 'CODE']
    cond_df = statis_data[statis_data.iloc[:, 0] == isic4_code].dropna()
    if not cond_df.empty:
        temp.append(cond_df.iloc[:, 4].unique().tolist())
    else:
        temp.append([])

for i in range(dup_text_data.shape[0]):
    dup_text_data.loc[i, 'HS_6'] = ', '.join(temp[i])

text_data['HS_6'] = ''
for i in range(dup_text_data.shape[0]):
    code = dup_text_data.loc[i, 'CODE']
    hs2017 = dup_text_data.loc[i, 'HS_6']
    text_data.loc[text_data['CODE'] == code, 'HS_6'] = hs2017

file_path = './data/비식별된 해외기업별 영문 텍스트데이터 2.csv'
text_data.to_csv(file_path, index=False)

print(f'> "{file_path}" already exists.')


# # 부류 목록 직접 수집 (관세율표-부류목록)
b01 = '살아 있는 동물과 동물성 생산품'
r01 = ['살아 있는 동물', 
        '육과 식용 설육(屑肉)', 
        '어류ㆍ갑각류ㆍ연체동물과 그 밖의 수생(水生) 무척추동물', 
        '낙농품, 새의 알, 천연꿀, 다른 류로 분류되지 않은 식용인 동물성 생산품', 
        '다른 류로 분류되지 않은 동물성 생산품']

b02 = '식물성 생산품'
r02 = ['살아 있는 수목과 그 밖의 식물, 인경(鱗莖)ㆍ뿌리와 이와 유사한 물품, 절화(切花)와 장식용 잎',
        '식용의 채소ㆍ뿌리ㆍ괴경(塊莖)',
        '식용의 과실과 견과류, 감귤류ㆍ멜론의 껍질',
        '커피ㆍ차ㆍ마테(maté)ㆍ향신료',
        '곡물',
        '제분공업의 생산품과 맥아, 전분, 이눌린(inulin), 밀의 글루텐(gluten)',
        '채유(採油)에 적합한 종자와 과실, 각종 종자와 과실, 공업용ㆍ의약용 식물, 짚과 사료용 식물',
        '락(lac), 검ㆍ수지ㆍ그 밖의 식물성 수액과 추출물(extract)',
        '식물성 편조물(編組物)용 재료와 다른 류로 분류되지 않은 식물성 생산품']

b03 = '동물성ㆍ식물성ㆍ미생물성 지방과 기름 및 이들의 분해생산물, 조제한 식용 지방과 동물성ㆍ식물성 왁스'
r03 = '동물성ㆍ식물성ㆍ미생물성 지방과 기름 및이들의 분해생산물, 조제한 식용 지방과 동물성ㆍ식물성 왁스'

b04 = '조제 식료품, 음료ㆍ주류ㆍ식초, 담배ㆍ제조한 담배 대용물, 연소시키지 않고 흡입하도록 만들어진 물품(니코틴을 함유하였는지에 상관없다), 니코틴을 함유한 그 밖의 물품으로 인체 내에 니코틴을 흡수시키도록 만들어진 것'
r04 = ['육류ㆍ어류ㆍ갑각류ㆍ연체동물이나 그 밖의 수생(水生) 무척추동물 또는 곤충의 조제품',
        '당류(糖類)와 설탕과자',
        '코코아와 그 조제품',
        '곡물ㆍ고운 가루ㆍ전분ㆍ밀크의 조제품과 베이커리 제품',
        '채소ㆍ과실ㆍ견과류나 식물의그 밖의 부분의 조제품',
        '각종 조제 식료품',
        '음료ㆍ주류ㆍ식초',
        '식품 공업에서 생기는 잔재물과 웨이스트(waste), 조제 사료',
        '담배와 제조한 담배 대용물, 연소시키지 않고흡입하도록 만들어진 물품(니코틴을 함유하였는지에 상관없다), 니코틴을 함유한 그 밖의 물품으로 인체 내에 니코틴을 흡수시키도록 만들어진 것']

b05 = '광물성 생산품'
r05 = ['소금, 황, 토석류(土石類), 석고ㆍ석회ㆍ시멘트',
        '광(鑛)ㆍ슬래그(slag)ㆍ회(灰)',
        '광물성 연료ㆍ광물유(鑛物油)와 이들의 증류물, 역청(瀝靑)물질, 광물성 왁스']

b06 = '화학공업이나 연관공업의 생산품'
r06 = ['무기화학품, 귀금속ㆍ희토류(稀土類)금속ㆍ방사성원소ㆍ동위원소의 유기화합물이나 무기화합물',
        '유기화학품',
        '의료용품',
        '비료',
        '유연용ㆍ염색용 추출물(extract), 탄닌과 이들의 유도체, 염료ㆍ안료와 그 밖의 착색제, 페인트ㆍ바니시(varnish), 퍼티(putty)와 그 밖의 매스틱(mastic), 잉크',
        '정유(essential oil)와 레지노이드(resinoid), 조제향료와 화장품ㆍ화장용품',
        '비누ㆍ유기계면활성제ㆍ조제 세제ㆍ조제 윤활제ㆍ인조 왁스ㆍ조제 왁스ㆍ광택용이나 연마용 조제품ㆍ양초와 이와 유사한 물품ㆍ조형용 페이스트(paste)ㆍ치과용 왁스와 플라스터(plaster)를 기본 재료로 한 치과용 조제품',
        '단백질계 물질, 변성전분, 글루(glue), 효소',
        '화약류, 화공품, 성냥, 발화성 합금, 특정 가연성 조제품',
        '사진용이나 영화용 재료',
        '각종 화학공업 생산품']

b07 = '플라스틱과 그 제품, 고무와 그 제품'
r07 = ['플라스틱과 그 제품',
        '고무와 그 제품']

b08 = '원피ㆍ가죽ㆍ모피와 이들의 제품, 마구, 여행용구ㆍ핸드백과 이와 유사한 용기, 동물 거트(gut)[누에의 거트(gut)는 제외한다]의 제품'
r08 = ['원피(모피는 제외한다)와 가죽',
        '가죽제품, 마구, 여행용구ㆍ핸드백과 이와 유사한 용기, 동물 거트(gut)[누에의 거트(gut)는 제외한다]의 제품',
        '모피ㆍ인조모피와 이들의 제품']

b09 = '목재와 그 제품, 목탄, 코르크와 그 제품, 짚ㆍ에스파르토(esparto)나 그 밖의 조물 재료의 제품, 바구니 세공물(basketware)과 지조세공물(枝條細工物)'
r09 = ['목재와 그 제품, 목탄',
        '코르크(cork)와 그 제품',
        '짚ㆍ에스파르토(esparto)나 그 밖의 조물 재료의 제품, 바구니 세공물(basketware)과 지조세공물(枝條細工物)']

b10 = '목재나 그 밖의 섬유질 셀룰로오스재료의 펄프, 회수한 종이ㆍ판지[웨이스트(waste)와 스크랩(scrap)], 종이ㆍ판지와 이들의 제품'
r10 = ['목재나 그 밖의 섬유질 셀룰로오스재료의 펄프, 회수한 종이ㆍ판지[웨이스트(waste)와 스크랩(scrap)]',
        '종이와 판지, 제지용 펄프ㆍ종이ㆍ판지의 제품',
        '인쇄서적ㆍ신문ㆍ회화ㆍ그 밖의 인쇄물, 수제(手製)문서ㆍ타자문서ㆍ도면']

b11 = '방직용 섬유와 방직용 섬유의 제품'
r11 = ['견',
        '양모ㆍ동물의 부드러운 털이나 거친 털ㆍ말의 털로 만든 실과 직물',
        '면',
        '그 밖의 식물성 방직용 섬유, 종이실(paper yarn)과 종이실로 만든 직물',
        '인조필라멘트, 인조방직용 섬유재료의 스트립(strip)과 이와 유사한 것',
        '인조스테이플섬유',
        '워딩(wadding)ㆍ펠트(felt)ㆍ부직포, 특수사, 끈ㆍ배의 밧줄(cordage)ㆍ로프ㆍ케이블과 이들의 제품',
        '양탄자류와 그 밖의 방직용 섬유로 만든 바닥깔개',
        '특수직물, 터프트(tuft)한 직물, 레이스, 태피스트리(tapestry), 트리밍(trimming), 자수천',
        '침투ㆍ도포ㆍ피복하거나 적층한 방직용 섬유의 직물, 공업용인 방직용 섬유제품',
        '메리야스 편물과 뜨개질 편물',
        '의류와 그 부속품(메리야스 편물이나 뜨개질 편물로 한정한다)',
        '의류와 그 부속품(메리야스 편물이나 뜨개질편물은 제외한다)',
        '제품으로 된 방직용 섬유의 그 밖의 물품, 세트, 사용하던 의류ㆍ방직용 섬유제품, 넝마']

b12 = '신발류ㆍ모자류ㆍ산류(傘類)ㆍ지팡이ㆍ시트스틱(seat-stick)ㆍ채찍ㆍ승마용 채찍과 이들의 부분품, 조제 깃털과 그 제품, 조화, 사람 머리카락으로 된 제품'
r12 = ['신발류ㆍ각반과 이와 유사한 것, 이들의 부분품',
        '모자류와 그 부분품',
        '산류(傘類)ㆍ지팡이ㆍ시트스틱(seat-stick)ㆍ채찍ㆍ승마용 채찍과 이들의 부분품',
        '조제 깃털ㆍ솜털과 그 제품, 조화, 사람 머리카락으로 된 제품']

b13 = '돌ㆍ플라스터(plaster)ㆍ시멘트ㆍ석면ㆍ운모나 이와 유사한 재료의 제품, 도자제품, 유리와 유리제품'
r13 = ['돌ㆍ플라스터(plaster)ㆍ시멘트ㆍ석면ㆍ운모나 이와 유사한 재료의 제품',
        '도자제품',
        '유리와 유리제품']

b14 = '천연진주ㆍ양식진주ㆍ귀석ㆍ반귀석ㆍ귀금속ㆍ귀금속을 입힌 금속과 이들의 제품, 모조 신변장식용품, 주화'
r14 = '천연진주ㆍ양식진주ㆍ귀석ㆍ반귀석ㆍ귀금속ㆍ귀금속을 입힌 금속과 이들의 제품, 모조 신변장식용품, 주화'

b15 = '비금속(卑金屬)과 그 제품'
r15 = ['철강',
        '철강의 제품',
        '구리와 그 제품',
        '니켈과 그 제품',
        '알루미늄과 그 제품',
        # '(유 보)',
        '납과 그 제품',
        '아연과 그 제품',
        '주석과 그 제품',
        '그 밖의 비금속(卑金屬), 서멧(cermet), 이들의 제품',
        '비금속(卑金屬)으로 만든 공구ㆍ도구ㆍ칼붙이ㆍ스푼ㆍ포크, 이들의 부분품',
        '비금속(卑金屬)으로 만든 각종 제품']

b16 = '기계류ㆍ전기기기와 이들의 부분품, 녹음기ㆍ음성재생기ㆍ텔레비전의 영상과 음향의 기록기ㆍ재생기와 이들의 부분품ㆍ부속품'
r16 = ['원자로ㆍ보일러ㆍ기계류와 이들의 부분품',
        '전기기기와 그 부분품, 녹음기ㆍ음성 재생기ㆍ텔레비전의 영상과 음성의 기록기ㆍ재생기와 이들의 부분품ㆍ부속품']

b17 = '차량ㆍ항공기ㆍ선박과 수송기기 관련품'
r17 = ['철도용이나 궤도용 기관차ㆍ차량과 이들의 부분품, 철도용이나 궤도용 장비품과 그 부분품, 기계식(전기기계식을 포함한다) 각종 교통신호용 기기',
        '철도용이나 궤도용 외의 차량과 그 부분품ㆍ부속품',
        '항공기와 우주선, 이들의 부분품',
        '선박과 수상 구조물']

b18 = '광학기기ㆍ사진용 기기ㆍ영화용 기기ㆍ측정기기ㆍ검사기기ㆍ정밀기기ㆍ의료용 기기, 시계, 악기, 이들의 부분품과 부속품'
r18 = ['광학기기ㆍ사진용 기기ㆍ영화용 기기ㆍ측정기기ㆍ검사기기ㆍ정밀기기ㆍ의료용 기기, 이들의 부분품과 부속품',
        '시계와 그 부분품',
        '악기와 그 부분품과 부속품']

b19 = '무기ㆍ총포탄과 이들의 부분품과 부속품'
r19 = '무기ㆍ총포탄과 이들의 부분품과 부속품'

b20 = '잡품'
r20 = ['가구, 침구ㆍ매트리스ㆍ매트리스 서포트(mattress support)ㆍ쿠션과 이와 유사한 물품, 다른 류로 분류되지 않은 조명기구, 조명용 사인ㆍ조명용 네임플레이트(name-plate)와 이와 유사한 물품, 조립식 건축물',
        '완구ㆍ게임용구ㆍ운동용구와 이들의 부분품과 부속품',
        '잡품']

b21 = '예술품ㆍ수집품ㆍ골동품'
r21 = '예술품ㆍ수집품ㆍ골동품'


# # 크롤링 데이터와 부류목록으로 HSCODE 자릿수 기준 데이터프레임 생성

# 크롤링 데이터는 관세법령정보포털 - 관세율표 - 2024년에 나오는 해설서 수집
hscode = pd.read_csv('./data/crawl_HSCODE.csv', dtype=str, encoding='utf-8-sig')
hscode.columns = ['HS_4', 'HS_6', 'HS_10', 'KOR', 'ENG', 'other0', 'other1', 'other2']
hscode = hscode.sort_values(by=['HS_4', 'HS_6', 'HS_10']).reset_index(drop=True)

# HS 2자리에 해당하는 부류 목록
temp = pd.DataFrame(hscode['HS_4'].apply(lambda x: x[:2]).drop_duplicates().reset_index(drop=True))
temp.columns = ['HS_2']
temp['BU'] = ''
temp['RYU'] = ''

# 부에 해당하는 류 리스트들을 cond에 직접 넣었음
# cond = [
# '97'
# ]
# idx = temp[temp['HS_2'].isin(cond)].index
# temp.loc[idx, 'BU'] = b21
# temp.loc[idx, 'RYU'] = r21
# temp.loc[idx]
hs2 = temp

# HS 4자리에 해당하는 국문과 영문 해설
is_nan_hs6 = pd.isnull(hscode['HS_6'])
is_nan_hs10 = pd.isnull(hscode['HS_10'])
temp = hscode[is_nan_hs6 & is_nan_hs10]
hs4 = temp[['HS_4', 'KOR', 'ENG']].reset_index(drop=True)

# HS 5자리에 해당하는 국문과 영문 해설
len_one_hs6 = hscode[hscode['HS_6'].str.len() == 1]
hs5 = len_one_hs6[['HS_4', 'HS_6', 'KOR', 'ENG']].reset_index(drop=True)
hs5.columns = ['HS_4', 'HS_5', 'KOR', 'ENG']

# HS 6자리에 해당하는 국문과 영문 해설
len_two_hs6 = hscode[hscode['HS_6'].str.len() == 2]
is_nan_hs10 = pd.isnull(len_two_hs6['HS_10'])
temp = len_two_hs6[is_nan_hs10]
hs6 = temp[['HS_4', 'HS_6', 'KOR', 'ENG']].reset_index(drop=True)

# HS 8자리에 해당하는 국문과 영문 해설
len_two_hs6 = hscode[(hscode['HS_6'].str.len() == 2) & (hscode['HS_10'].str.len() == 2)]
not_nan_hs4 = pd.notnull(len_two_hs6['HS_4'])
not_nan_hs6 = pd.notnull(len_two_hs6['HS_6'])
not_nan_hs10 = pd.notnull(len_two_hs6['HS_10'])
temp = len_two_hs6[not_nan_hs4 & not_nan_hs6 & not_nan_hs10]
hs8 = temp[['HS_4', 'HS_6', 'HS_10', 'KOR', 'ENG']].reset_index(drop=True)
hs8.columns = ['HS_4', 'HS_6', 'HS_8', 'KOR', 'ENG']

# HS 10자리에 해당하는 국문과 영문 해설
len_two_hs6 = hscode[hscode['HS_10'].str.len() == 4]
not_nan_hs4 = pd.notnull(len_two_hs6['HS_4'])
not_nan_hs6 = pd.notnull(len_two_hs6['HS_6'])
not_nan_hs10 = pd.notnull(len_two_hs6['HS_10'])
temp = len_two_hs6[not_nan_hs4 & not_nan_hs6 & not_nan_hs10]
hs10 = temp[['HS_4', 'HS_6', 'HS_10', 'KOR', 'ENG']].reset_index(drop=True)

# 가장 우측에 ':' 제거 
dfs = [hs4, hs5, hs6, hs8, hs10]
lans = ['KOR', 'ENG']
for df in dfs:
    for lan in lans:
        for idx in range(df.shape[0]):
            if df.loc[idx, lan][-1] == ':':
                df.loc[idx, lan] = df.loc[idx, lan].rstrip(':').strip()

def save_csv(df, file_path):
    if os.path.exists(file_path):
        print(f'> "{file_path}" already exists.')
    else:
        df.to_csv(file_path, index=False, encoding='utf-8')

save_csv(hs2, './data/HS_2.csv')
save_csv(hs4, './data/HS_4.csv')
save_csv(hs5, './data/HS_5.csv')
save_csv(hs6, './data/HS_6.csv')
save_csv(hs8, './data/HS_8.csv')
save_csv(hs10, './data/HS_10.csv')

print('> done.')