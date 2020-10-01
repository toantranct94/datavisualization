import pandas as pd 
import os 
import unicodedata
from unidecode import unidecode

def convert_type(type_name):
    if type_name == 'string':
        return 'STRING'

def danso(filename='danso1'):
    number_of_feature = 3
    number_of_year = 3
    path = os.path.join(os.getcwd(), 'dataset', '{}.csv'.format(filename))
    df_danso = pd.read_csv(path, skiprows=1, sep=';')
    df_danso = df_danso.dropna()
    # print(df_danso)
    # print(df_danso.iloc[0,:])
    rows = df_danso.iloc[0,:].values
    rows = [x.strip() for x in rows]
    rows = rows[1:]
    print(rows)

    places = df_danso.iloc[:,0].values
    places = [x.strip() for x in places]
    places = places[1:]
    print(places)


    cols = df_danso.columns.tolist()[1:]
    cols = cols[::number_of_year]
    print(cols)
    results = []
    datatypes = ['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']
    results.append(datatypes)
    for index, place in enumerate(places):
        vals = df_danso.iloc[index + 1,:].values
        vals = [x if x != '..' else '0' for x in vals]
        # print(vals)
        # place = unicodedata.normalize('NFKD', place).encode('ascii', 'ignore')
        place = unidecode(place)
        for i in range(number_of_year):
            try:
                results.append([place.replace(' ', '_'), str(rows[i]), vals[i + 1 + 3 * 0], vals[i + 1 + 3 * 1], vals[i + 1 + 3 * 2], vals[i + 1 + 3 * 3], vals[i + 1 + 3 * 4]])
            except:
                results[0] = ['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']
                results.append([place.replace(' ', '_'), str(rows[i]), vals[i + 1 + 3 * 0], vals[i + 1 + 3 * 1], vals[i + 1 + 3 * 2]])
                    # print(results)    
    labels = ['1', '2'] + cols
    df = pd.DataFrame(results, columns=labels)
    print(df.head())
    
    df.to_csv(os.path.join(os.getcwd(), 'dataset', '{}.tm3'.format(filename)), sep='\t', index=False)

if __name__ == '__main__':
    danso(filename='danso2')
    pass