import pandas as pd 
import os 
import unicodedata
from unidecode import unidecode
import numpy as np
from itertools import repeat

def read_csv_file(path):
    df = pd.read_csv(path, skiprows=1, sep=';')
    df = df.dropna()
    return df

def danso(filename='danso_diaphuong_nam_chitieu', number_of_year=3):
    
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'DanSo', '{}.csv'.format(filename))
    df_danso = read_csv_file(path=path)

    rows = df_danso.iloc[0,:].values
    rows = [x.strip() for x in rows]
    rows = rows[1:]

    places = df_danso.iloc[:,0].values
    places = [x.strip() for x in places]
    places = places[1:]

    cols = df_danso.columns.tolist()[1:]
    cols = cols[::number_of_year]
    number_of_feature = len(cols)
    cols = [unidecode(x) for x in cols]
    results = []
    datatypes = ['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']
    datatypes = datatypes[:number_of_feature]
    results.append(datatypes)
    for index, place in enumerate(places):
        vals = df_danso.iloc[index + 1,:].values
        vals = [x if x != '..' else '0' for x in vals]
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
    
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'DanSo', '{}.tm3'.format(filename)), sep='\t', index=False)

def laodong(filename='laodong_15t_cachtinh_nhomtuoi_nam', number_of_year=3):
    
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'LaoDong', '{}.csv'.format(filename))
    df_laodong = read_csv_file(path=path)
    cols = df_laodong.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']

    df_laodong = df_laodong.iloc[::-1]
    df_laodong = df_laodong.values.tolist()
    df_laodong = [x[1:] for x in df_laodong]
    df_laodong.append(datatypes)
    df_laodong = pd.DataFrame(df_laodong, columns=cols)
    df_laodong = df_laodong.iloc[::-1]
    df_laodong.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'LaoDong', '{}.tm3'.format(filename)), sep='\t', index=False)

def giaoduc(filename='giaoduc_chitieu_nam', number_of_year=3):

    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'GiaoDuc', '{}.csv'.format(filename))
    df_giaoduc = read_csv_file(path=path)

    cols = df_giaoduc.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['1'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']

    df_giaoduc[df_giaoduc.columns[0]] = df_giaoduc[df_giaoduc.columns[0]].apply(unidecode)
    df_giaoduc = df_giaoduc.iloc[::-1]
    df_giaoduc = df_giaoduc.values.tolist()
    df_giaoduc.append(datatypes)
    df_giaoduc = pd.DataFrame(df_giaoduc, columns=cols)
    df_giaoduc = df_giaoduc.iloc[::-1]
    df_giaoduc.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'GiaoDuc', '{}.tm3'.format(filename)), sep='\t', index=False)


def congnghiep(filename='congnghiep'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'CongNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')

    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::5]
    features = [x for item in features for x in repeat(item, 4)]
    df = df.dropna()
    df = df.values.tolist()
    print(len(df))

    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) for x in val]

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'CongNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)


if __name__ == '__main__':
    # danso(filename='danso_diaphuong_nam_chitieu')
    # danso(filename='danso_diaphuong_thanhthi_nongthon')
    # laodong(filename='laodong_15t_cachtinh_nhomtuoi_nam')
    # laodong(filename='laodong_15t_gioitinh_thanhthi_nongthon_cachtinh_nam_phanto')
    # giaoduc(filename='giaoduc_chitieu_nam')
    congnghiep(filename='congnghiep')
    