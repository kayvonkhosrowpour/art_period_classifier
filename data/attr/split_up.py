import pandas as pd

csv = '/Users/kayvon/code/divp/proj/data/data_table/data_table.csv'
frame = pd.read_csv(csv)
frame['who'] = None


frame.loc[(frame.index < len(frame.index)//2), 'who'] = 'Kayvon'
frame.loc[(frame.index >= len(frame.index)//2), 'who'] = 'Allison'

frame.to_csv('/Users/kayvon/code/divp/proj/data/data_table/data_table_who.csv', sep=',', encoding='utf-8')
