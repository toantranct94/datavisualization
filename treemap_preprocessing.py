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

def danso(filename='Dân số trung bình phân theo địa phương, giới tính và thành thị nông thôn', number_of_year=3):
    
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'DanSo', '{}.csv'.format(filename))
    df = read_csv_file(path=path)

    rows = df.iloc[0,:].values
    rows = [x.strip() for x in rows]
    rows = rows[1:]

    places = df.iloc[:,0].values
    places = [x.strip() for x in places]
    places = places[1:]

    cols = df.columns.tolist()[1:]
    cols = cols[::number_of_year]
    number_of_feature = len(cols)
    cols = [unidecode(x) for x in cols]
    results = []
    datatypes = ['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']
    # datatypes = datatypes[:number_of_feature]
    results.append(datatypes)
    for index, place in enumerate(places):
        vals = df.iloc[index + 1,:].values
        vals = [x if x != '..' else '0' for x in vals]
        place = unidecode(place)
        for i in range(number_of_year):
            try:
                results.append([place.replace(' ', '_'), str(rows[i]), vals[i + 1 + 3 * 0], vals[i + 1 + 3 * 1], vals[i + 1 + 3 * 2], vals[i + 1 + 3 * 3], vals[i + 1 + 3 * 4]])
            except:
                results[0] = ['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']
                results.append([place.replace(' ', '_'), str(rows[i]), vals[i + 1 + 3 * 0], vals[i + 1 + 3 * 1], vals[i + 1 + 3 * 2]])
                    # print(results)    
    results = [x if x != '..' else 0 for x in results]
    labels = ['Vung', 'Nam'] + cols
    df = pd.DataFrame(results, columns=labels)
    
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'DanSo', '{}.tm3'.format(filename)), sep='\t', index=False)

def laodong(filename='laodong_15t_cachtinh_nhomtuoi_nam', number_of_year=3):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'LaoDong', '{}.csv'.format(filename))
    # df = read_csv_file(path=path)
    df = pd.read_csv(path, skiprows=1, sep=';')
    if filename == 'Lực lượng lao động từ 15 tuổi trở lên phân theo giới tính và phân theo thành thị, nông thôn':
        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
    else:
        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
    number_of_feature = 4
    cols = df.columns.values[1:].tolist()
    cols = ['Tong so/Co cau', 'Nam'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::number_of_feature]
    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    features = features[:-3]
    df = df.dropna()
    df = df.values.tolist()
    if len(features) == len(df):
        print("CHECK")
    
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
        df[index] = [x if x != '..' else 0 for x in df[index]]
    # datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)


    # cols = df.columns.values[1:].tolist()
    # cols = [unidecode(x) for x in cols]
    # datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']

    # df = df.iloc[::-1]
    # df = df.values.tolist()
    # df = [x[1:] for x in df]
    # df.append(datatypes)
    # df = pd.DataFrame(df, columns=cols)
    # df = df.iloc[::-1]
    print(df)

    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'DanSo_LaoDong', 'LaoDong', '{}.tm3'.format(filename)), sep='\t', index=False)

def giaoduc(filename='giaoduc_chitieu_nam', number_of_year=3):

    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'GiaoDuc', '{}.csv'.format(filename))
    df = read_csv_file(path=path)

    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['1'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']

    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'GiaoDuc', '{}.tm3'.format(filename)), sep='\t', index=False)

def congnghiep(filename='congnghiep'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'CongNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    cols = df.columns.values[1:].tolist()
    cols = ['San pham', 'Thanh phan KT'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::5]
    features = [x for item in features for x in repeat(item, 4)]
    df = df.dropna()
    df = df.values.tolist()
    print(len(df))

    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if x != '..' else '0' for x in val]

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'CongNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)

def dautu(filename='vondautu_thanhphankt_cachtinh_nam'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'DauTu', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')

    cols = df.columns.values[1:].tolist()
    cols = ['Cach tinh', 'Nam'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::4]
    features = [x for item in features for x in repeat(item, 3)]
    df = df.dropna()
    df = df.values.tolist()
    
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
        df[index] = [x if x != '..' else '0' for x in df[index]]

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'DauTu', '{}.tm3'.format(filename)), sep='\t', index=False)

def xaydung(filename='dientich_xaydung_loainha_nam'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'XayDung', '{}.csv'.format(filename))
    df = read_csv_file(path=path)
    # print(df)
    if filename == 'Diện tích sàn xây dựng nhà ở hoàn thành trong năm phân theo loại nhà':
        typed = ['Loai nha'] 
    else:
        typed = ['Vung'] 
    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = typed + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']

    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df = [x if x != '..' else 0 for x in df]
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'XayDung', '{}.tm3'.format(filename)), sep='\t', index=False)

def lamnghiep(filename='Diện tích rừng trồng mới tập trung phân theo loại rừng'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'LamNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    print(df)
    if filename == 'Diện tích rừng trồng mới tập trung phân theo loại rừng':
        number_of_feature = 4
    else:
        number_of_feature = 8
        
    cols = df.columns.values[1:].tolist()
    cols = ['1', 'Nam'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::number_of_feature]

    if number_of_feature == 8:
        features = features[:-1]

    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    df = df.dropna()
    df = df.values.tolist()
    
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
        df[index] = [x if x != '..' else '0' for x in df[index]]

    try:
        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
        df = datatypes + df
        df = pd.DataFrame(df, columns=cols)
    except:
        df = df[1:]
        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
        df = datatypes + df
        df = pd.DataFrame(df, columns=cols)
    
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'LamNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)

def nongnghiep(filename='Diện tích và sản lượng lúa cả năm'):
    if filename == 'Diện tích và sản lượng lúa cả năm':
        number_of_feature = 4
    else:
        number_of_feature = 13

    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'NongNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    print(df)
    cols = df.columns.values[1:].tolist()
    cols = ['Gia tri/Chi so phat trien', 'Nam'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::number_of_feature]
    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    df = df.dropna()
    df = df.values.tolist()
    
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
        df[index] = [x if x != '..' else '0' for x in df[index]]

    try:
        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
        df = datatypes + df
        df = pd.DataFrame(df, columns=cols)
    except:
        df = df[1:]
        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
        df = datatypes + df
        df = pd.DataFrame(df, columns=cols)
        pass
    
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'NongNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)

def nongnghiep1(filename='Sản lượng sản phẩm chăn nuôi chủ yếu'):
    
    number_of_feature = 4
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'NongNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['San luong'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']
    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'NongNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)

def thuysan(filename='Diện tích nuôi trồng thủy sản'):
    
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'ThuySan', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['Dien tich'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']
    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df = [x if x != '..' else 0 for x in df]
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'ThuySan', '{}.tm3'.format(filename)), sep='\t', index=False)

def thuysan1(filename='Sản lượng thuỷ sản'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'ThuySan', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    number_of_feature = 4
    cols = df.columns.values[1:].tolist()
    cols = ['San luong', 'Nam'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::number_of_feature]
    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    df = df.dropna()
    df = df.values.tolist()
    df = [x if x != '..' else 0 for x in df]
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'ThuySan', '{}.tm3'.format(filename)), sep='\t', index=False)

def cosokinhte(filename='Lao động trong các cơ sở kinh tế cá thể phi nông nghiệp'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'DoanhNghiep_HopTacXa_CoSoKinhTeCaThePhiNongNghiep', 'CoSoKinhTeCaThePhiNongNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['Nganh kinh te'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']
    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df = [x if x != '..' else '0' for x in df]
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    # print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'DoanhNghiep_HopTacXa_CoSoKinhTeCaThePhiNongNghiep', 'CoSoKinhTeCaThePhiNongNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)

def doanhnghiep(filename='Doanh thu thuần sản xuất kinh doanh của các doanh nghiệp đang hoạt động có kết quả sản xuất kinh doanh phân theo loại hình doanh'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'DoanhNghiep_HopTacXa_CoSoKinhTeCaThePhiNongNghiep', 'DoanhNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    # print(df)
    number_of_feature = 14
    cols = df.columns.values[1:].tolist()
    cols = ['Don vi', 'Loai hinh'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::number_of_feature]
    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    df = df.dropna()
    df = df.values.tolist()
    # print(df)
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        val = [float(x.replace(',', '.')) if val.index(x) >= 2 else x for x in val]
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
        df[index] = [x if x != '..' else 0 for x in df[index]]

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)
    # print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'DoanhNghiep_HopTacXa_CoSoKinhTeCaThePhiNongNghiep', 'DoanhNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)

def hoptacxa(filename='Số hợp tác xã phân theo địa phương'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'DoanhNghiep_HopTacXa_CoSoKinhTeCaThePhiNongNghiep', 'HopTacXa', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    # print(df)
    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['Dia phuong'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']
    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df = [x if x != '..' else 0 for x in df]
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    # print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'DoanhNghiep_HopTacXa_CoSoKinhTeCaThePhiNongNghiep', 'HopTacXa', '{}.tm3'.format(filename)), sep='\t', index=False)

def vantai(filename='Số lượt hành khách luân chuyển phân theo ngành vận tải'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'VanTai_BuuChinhVienThong', 'VanTai', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    # print(df)
    number_of_feature = 4
    cols = df.columns.values[1:].tolist()
    cols = ['1', 'Nam'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::number_of_feature]
    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    features = features[:-3]
    df = df.dropna()
    df = df.values.tolist()
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)
    # print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'VanTai_BuuChinhVienThong', 'VanTai', '{}.tm3'.format(filename)), sep='\t', index=False)

def buuchinh(filename='Doanh thu bưu chính, chuyển phát và viễn thông'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'VanTai_BuuChinhVienThong', 'BuuChinhVienThong', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    # print(df)
    number_of_feature = 4
    cols = df.columns.values[1:].tolist()
    cols = ['1', '2'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::number_of_feature]
    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    features = features[:-3]
    df = df.dropna()
    df = df.values.tolist()
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]
    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)
    # print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'VanTai_BuuChinhVienThong', 'BuuChinhVienThong', '{}.tm3'.format(filename)), sep='\t', index=False)

def moitruong(filename='Thiệt hại do thiên tai'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'MoiTruong', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    # print(df)
    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['1'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']
    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'MoiTruong', '{}.tm3'.format(filename)), sep='\t', index=False)

def moitruong1(filename='Xử lý chất thải rắn và nước thải của các khu công nghiệp'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'MoiTruong', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    years = df.columns.values
    years = years[2:]
    years = years[::3]
    number_of_feature = 6
    features = df.iloc[:,0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[1:]
    features = features[::number_of_feature]
    features = features[:-1]
    features = [x for item in features for x in repeat(item, number_of_feature - 1)]
    df = df.dropna()
    df = df.values.tolist()
    labels = df[0]
    labels = [unidecode(x) for x in labels]
    df = df[1:]
    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [unidecode(x) if type(x) is str else x for x in val]

    df = pd.DataFrame(df, columns=labels)
    print(df.head())

    rows = df.iloc[0,:].values
    rows = [x.strip() for x in rows]
    rows = rows[1:]

    places = df.iloc[:,0].values
    # places = [x.strip() for x in places]
    # places = places[1:]
    rows = labels[2:]
    number_of_year = 3
    cols = df.columns.tolist()[1:]
    number_of_feature = len(cols)
    results = []
    datatypes = ['STRING', 'STRING', 'INTEGER', 'FLOAT', 'FLOAT', 'FLOAT']
    results.append(datatypes)
    for index, place in enumerate(places):
        vals = df.iloc[index,:].values
        vals = [x if x != '..' else '0' for x in vals]
        for i in range(number_of_year):
            results.append([place + ' ' + rows[i], vals[1], years[i] ,vals[i + 2 + 3 * 0], vals[i + 2 + 3 * 1], vals[i + 2 + 3 * 2]])
    columns = ['1', '2', '3'] + list(set(rows))
    df = pd.DataFrame(results, columns=columns)
    print(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'MoiTruong', '{}.tm3'.format(filename)), sep='\t', index=False)

def dulich(filename='ketqua_kinhdoanh_nganhdulich'):
    path = os.path.join(os.getcwd(), 'Data', 'ThuongMai_DuLich', 'DuLich', '{}.csv'.format(filename))

    if (filename == 'ketqua_kinhdoanh_nganhdulich'):
        df = read_csv_file(path=path)
        print(df)
        cols = df.columns.values[1:].tolist()
        cols = [unidecode(x) for x in cols]
        cols = ['1'] + cols
        datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']

        df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
        df = df.iloc[::-1]
        df = df.values.tolist()
        df.append(datatypes)
        df = pd.DataFrame(df, columns=cols)
        df = df.iloc[::-1]
    else:
        df = pd.read_csv(path, skiprows=1, sep=';')
        cols = df.columns.values[1:].tolist()
        cols = ['1', '2'] + cols[1:]
        cols = [unidecode(x) for x in cols]
        features = df.iloc[:, 0].values.tolist()
        features = [unidecode(x) for x in features]
        print(features)
        features = features[::8]
        features = [x for item in features for x in repeat(item, 7)]
        print(features)
        df = df.dropna()
        df = df.values.tolist()

        for index, (feature, val) in enumerate(zip(features, df)):
            val[0] = feature
            df[index] = [(0 if (x == '..') else unidecode(x)) if type(x) is str else x for x in val]

        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
        df = datatypes + df
        df = pd.DataFrame(df, columns=cols)
    df.to_csv(os.path.join(os.getcwd(), 'Data', 'ThuongMai_DuLich', 'DuLich', '{}.tm3'.format(filename)), sep='\t', index=False)

def thuongmai(filename='tri_gia_xuat_khau_hang_hoa_theo_khu_vuc_kinh_te_nhom_hang'):
    path = os.path.join(os.getcwd(), 'Data', 'ThuongMai_DuLich', 'ThuongMai', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')

    if (filename == 'tytrong_giatri_xuatkhau_nhapkhau_tongsanpham_trongnuoc'):
        cols = df.columns.values[1:].tolist()
        cols = [unidecode(x) for x in cols]
        cols = ['1'] + cols
        datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']

        df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
        df = df.iloc[::-1]
        df = df.values.tolist()
        df.append(datatypes)
        df = pd.DataFrame(df, columns=cols)
        df = df.iloc[::-1]
    else:
        cols = df.columns.values[1:].tolist()
        cols = ['1', '2'] + cols[1:]
        cols = [unidecode(x) for x in cols]
        features = df.iloc[:, 0].values.tolist()
        features = [unidecode(x) for x in features]
        features = features[::13]
        features = [x for item in features for x in repeat(item, 12)]
        df = df.dropna()
        df = df.values.tolist()

        for index, (feature, val) in enumerate(zip(features, df)):
            val[0] = feature
            df[index] = [(0 if (x == '..') else unidecode(x)) if type(x) is str else x for x in val]

        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
        df = datatypes + df
        df = pd.DataFrame(df, columns=cols)

    df.to_csv(os.path.join(os.getcwd(), 'Data', 'ThuongMai_DuLich', 'ThuongMai', '{}.tm3'.format(filename)),
                  sep='\t', index=False)

def trattuantoanxahoi(filename='trattu_antoan_xahoi'):
    path = os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'TratTuAnToanXaHoi', '{}.csv'.format(filename))

    df = read_csv_file(path=path)
    cols = df.columns.values[1:].tolist()
    cols = [unidecode(x) for x in cols]
    cols = ['1'] + cols
    datatypes = ['STRING', 'FLOAT', 'FLOAT', 'FLOAT']

    df[df.columns[0]] = df[df.columns[0]].apply(unidecode)
    df = df.iloc[::-1]
    df = df.values.tolist()
    df.append(datatypes)
    df = pd.DataFrame(df, columns=cols)
    df = df.iloc[::-1]
    df.to_csv(os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'TratTuAnToanXaHoi', '{}.tm3'.format(filename)), sep='\t', index=False)

def thethao(filename='so_huy_chuong_the_thao'):
    path = os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'TheThao', '{}.csv'.format(filename))

    df = pd.read_csv(path, skiprows=1, sep=';')

    cols = df.columns.values[1:].tolist()
    cols = ['1', '2'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:, 0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::4]
    features = [x for item in features for x in repeat(item, 3)]
    df = df.dropna()
    df = df.values.tolist()

    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [(0 if (x == '..') else unidecode(x)) if type(x) is str else x for x in val]

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)

    df.to_csv(os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'TheThao', '{}.tm3'.format(filename)), sep='\t', index=False)

def vanhoa(filename='xuat_ban_sach_van_hoa_pham_bao_tap_chi'):
    path = os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'VanHoa', '{}.csv'.format(filename))

    df = pd.read_csv(path, skiprows=1, sep=';')

    cols = df.columns.values[1:].tolist()
    cols = ['1', '2'] + cols[1:]
    cols = [unidecode(x) for x in cols]
    features = df.iloc[:, 0].values.tolist()
    features = [unidecode(x) for x in features]
    features = features[::23]
    features = [x for item in features for x in repeat(item, 22)]
    df = df.dropna()
    df = df.values.tolist()

    for index, (feature, val) in enumerate(zip(features, df)):
        val[0] = feature
        df[index] = [(0 if (x == '..') else unidecode(x)) if type(x) is str else x for x in val]

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)

    df.to_csv(os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'VanHoa', '{}.tm3'.format(filename)), sep='\t', index=False)

def yte(filename='so_nhan_luc_y_te'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'Yte', '{}.csv'.format(filename))

    df = pd.read_csv(path, skiprows=1, sep=';')

    if (filename == 'so_co_so_kham_chua_benh_co_so_cap_quan_ly'):
        df = df.dropna()
        print(df)
        rows = df.iloc[0,:].values
        rows = [x.strip() for x in rows]
        rows = rows[1:]
        years = df.columns.values.tolist()
        years = years[1:][::4]
        places = df.iloc[:,0].values
        places = [x.strip() for x in places]
        places = places[1:]
        number_of_year = 3
        cols = df.columns.tolist()[1:]
        number_of_feature = len(cols)
        results = []
        datatypes = ['STRING', 'INTEGER', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']
        results.append(datatypes)
        for index, place in enumerate(places):
            vals = df.iloc[index + 1,:].values
            vals = [x if x != '..' else '0' for x in vals]
            for i in range(number_of_year):
                results.append([unidecode(place), years[i] ,vals[i + 1 + 3 * 0], vals[i + 1 + 3 * 1], vals[i + 1 + 3 * 2], vals[i + 1 + 3 * 3]])
        columns = ['Cap quan ly', 'Nam'] + list(set(rows))
        columns = [unidecode(x) for x in columns]
        df = pd.DataFrame(results, columns=columns)
        print(df) 
        
        pass
    else:
        cols = df.columns.values[1:].tolist()
        cols = ['1', '2'] + cols[1:]
        cols = [unidecode(x) for x in cols]
        features = df.iloc[:, 0].values.tolist()
        features = [unidecode(x) for x in features]
        features = features[::9]
        features = [x for item in features for x in repeat(item, 8)]
        df = df.dropna()
        df = df.values.tolist()

        for index, (feature, val) in enumerate(zip(features, df)):
            val[0] = feature
            df[index] = [(0 if (x == '..') else unidecode(x)) if type(x) is str else x for x in val]

        datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
        df = datatypes + df
        df = pd.DataFrame(df, columns=cols)

    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'Yte', '{}.tm3'.format(filename)), sep='\t', index=False)

def mucsongdancu(filename='Chi tiêu bình quân đầu người một tháng theo giá hiện hành theo khoản chi, theo thành thị, nông thôn và theo vùng'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'MucSongDanCu', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    rows = df.iloc[0,:].values
    rows = [x.strip() for x in rows]
    rows = rows[1:]
    years = df.columns.values.tolist()
    years = years[1:][::5]
    places = df.iloc[:,0].values
    places = [x.strip() for x in places]
    places = places[1:]
    number_of_year = 3
    cols = df.columns.tolist()[1:]
    number_of_feature = len(cols)
    results = []
    datatypes = ['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']
    results.append(datatypes)
    for index, place in enumerate(places):
        vals = df.iloc[index + 1,:].values
        vals = [x if x != '..' else '0' for x in vals]
        for i in range(number_of_year):
            results.append([unidecode(place), unidecode(years[i]) ,vals[i + 1 + 3 * 0], vals[i + 1 + 3 * 1], vals[i + 1 + 3 * 2], vals[i + 1 + 3 * 3]])
    columns = ['1'] + list(set(rows))
    columns = [unidecode(x) for x in columns]
    df = pd.DataFrame(results, columns=columns)
    print(df) 
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'MucSongDanCu', '{}.tm3'.format(filename)), sep='\t', index=False)


if __name__ == '__main__':
    # danso(filename='Dân số trung bình phân theo địa phương, giới tính và thành thị nông thôn')
    # danso(filename='Diện tích, dân số và mật độ dân số phân theo địa phương')
    # laodong(filename='Lực lượng lao động từ 15 tuổi trở lên phân theo giới tính và phân theo thành thị, nông thôn')
    # laodong(filename='Lực lượng lao động từ 15 tuổi trở lên phân theo nhóm tuổi')
    # giaoduc(filename='Giáo dục nghề nghiệp')
    # congnghiep(filename='Sản phẩm chủ yếu của ngành công nghiệp phân theo thành phần kinh tế')
    # dautu(filename='Vốn đầu tư phát triển toàn xã hội thực hiện phân theo thành phần kinh tế')
    # xaydung(filename='Diện tích sàn xây dựng nhà ở hoàn thành trong năm phân theo loại nhà')
    # xaydung(filename='Diện tích sàn xây dựng nhà ở hoàn thành trong năm phân theo vùng')
    # lamnghiep(filename='Diện tích rừng trồng mới tập trung phân theo loại rừng')
    # lamnghiep(filename='Sản lượng gỗ phân theo loại hình kinh tế')
    # nongnghiep(filename='Diện tích và sản lượng lúa cả năm')
    # nongnghiep(filename='Sản lượng một số cây lâu năm')
    # nongnghiep1(filename='Sản lượng sản phẩm chăn nuôi chủ yếu')
    # thuysan(filename='Diện tích nuôi trồng thủy sản')
    # thuysan1(filename='Sản lượng thuỷ sản')
    # cosokinhte(filename='Lao động trong các cơ sở kinh tế cá thể phi nông nghiệp')
    # cosokinhte(filename='Số cơ sở kinh tế cá thể phi nông nghiệp phân theo ngành kinh tế')
    # doanhnghiep(filename='Doanh thu thuần sản xuất kinh doanh của các doanh nghiệp đang hoạt động có kết quả sản xuất kinh doanh phân theo loại hình doanh')
    # doanhnghiep(filename='Số doanh nghiệp đang hoạt động có kết quả sản xuất kinh doanh tại thời điểm 3112 hàng năm phân theo loại hình doanh nghiệp')
    # doanhnghiep(filename='Tổng số lao động trong các doanh nghiệp đang hoạt động có kết quả sản xuất kinh doanh tại thời điểm 3112 hàng năm phân theo loại hình doanh')
    # hoptacxa(filename='Số hợp tác xã phân theo địa phương')
    # hoptacxa(filename='Số lao động trong hợp tác xã phân theo địa phương')
    vantai(filename='Số lượt hành khách luân chuyển phân theo ngành vận tải')
    vantai(filename='Số lượt hành khách vận chuyển phân theo ngành vận tải')
    # buuchinh(filename='Doanh thu bưu chính, chuyển phát và viễn thông')
    # moitruong(filename='Thiệt hại do thiên tai')
    # moitruong1(filename='Xử lý chất thải rắn và nước thải của các khu công nghiệp')
    # yte(filename='so_co_so_kham_chua_benh_co_so_cap_quan_ly')
    # mucsongdancu(filename='Chi tiêu bình quân đầu người một tháng theo giá hiện hành theo khoản chi, theo thành thị, nông thôn và theo vùng')
    # mucsongdancu(filename='Thu nhập bình quân đầu người một tháng theo giá hiện hành theo nguồn thu, theo thành thị, nông thôn, theo vùng')

    pass
    