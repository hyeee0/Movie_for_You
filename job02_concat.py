# 크롤링한 파일 하나로 합쳐주자

import pandas as pd
import glob
import datetime

data_path = glob.glob('./crawling_data_2/*.csv')
df = pd.DataFrame() # 빈데이터프레임 생성
for path in data_path:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace=True)
    df.drop_duplicates(inplace=True)  # 중복데이터 제거
    df = pd.concat([df, df_temp], ignore_index=True)

df.drop_duplicates(inplace=True)  # 중복데이터 제거
df.info()
print(df.titles.value_counts())
df.to_csv('./crawling_data_2/review_2016-2022.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)