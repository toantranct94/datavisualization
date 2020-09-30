import pandas as pd 
import os 

def danso():
    number_of_feature = 3
    path = os.path.join(os.getcwd(), 'dataset', 'danso1.csv')
    df_danso = pd.read_csv(path, skiprows=1, sep=';')
    df_danso = df_danso.dropna()
    print(df_danso)
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
    print(cols)
    results = []
    for index, place in enumerate(places):
        vals = df_danso.iloc[index + 1,:].values
        # print(vals)
        for i in range(number_of_feature):
            results.append([place,rows[i], vals[i + 1 + 3 * 0], vals[i + 1 + 3 * 1], vals[i + 1 + 3 * 2], vals[i + 1 + 3 * 3], vals[1 + 3 * 4]])
        print(results)    
        break


if __name__ == '__main__':
    danso()
    pass