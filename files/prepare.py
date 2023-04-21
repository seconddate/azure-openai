import pandas as pd

file_path = 'bus.csv'
df = pd.read_csv(file_path, encoding='cp949')
print(df.head())

lst = df.to_dict('records')

new_lst = []
for item in lst:
    new_lst.append({
        'prompt': item['정류장아이디'],
        'completion': item['정류장 명칭']
    })

df = pd.DataFrame(new_lst)
df.to_json('test.jsonl', orient='records', lines=True, force_ascii=False)
