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
    df_laodong.to_csv(os.path.join(os.getcwd(), 'Data', 'DanSo_LaoDong', 'LaoDong', '{}.tm3'.format(filename)), sep='\t', index=False)

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
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'GiaoDuc', '{}.tm3'.format(filename)), sep='\t', index=False)

def congnghiep(filename='congnghiep'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'CongNghiep', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')
    cols = df.columns.values[1:].tolist()
    cols = ['1', '2'] + cols[1:]
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
        df[index] = [unidecode(x) for x in val]

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(os.getcwd(), 'dataset', 'Data', 'CongNghiep_DauTu_XayDung', 'CongNghiep', '{}.tm3'.format(filename)), sep='\t', index=False)

def dautu(filename='vondautu_thanhphankt_cachtinh_nam'):
    path = os.path.join(os.getcwd(), 'Data', 'CongNghiep_DauTu_XayDung', 'DauTu', '{}.csv'.format(filename))
    df = pd.read_csv(path, skiprows=1, sep=';')

    cols = df.columns.values[1:].tolist()
    cols = ['1', '2'] + cols[1:]
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

    datatypes = [['STRING', 'STRING', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT']]
    df = datatypes + df
    df = pd.DataFrame(df, columns=cols)
    df.to_csv(os.path.join(os.getcwd(), 'Data', 'CongNghiep_DauTu_XayDung', 'DauTu', '{}.tm3'.format(filename)), sep='\t', index=False)

def xaydung(filename='dientich_xaydung_loainha_nam'):
    path = os.path.join(os.getcwd(), 'Data', 'CongNghiep_DauTu_XayDung', 'XayDung', '{}.csv'.format(filename))
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
    df.to_csv(os.path.join(os.getcwd(), 'Data', 'CongNghiep_DauTu_XayDung', 'XayDung', '{}.tm3'.format(filename)), sep='\t', index=False)

def lamnghiep(filename='lamnghiep_dientich_chiso_nam_loai'):
    path = os.path.join(os.getcwd(), 'dataset', 'Data', 'NongNghiep_LamNghiep_ThuySan', 'LamNghiep', '{}.csv'.format(filename))
    df = read_csv_file(path=path)
    print(df)

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
    path = os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'Yte', '{}.csv'.format(filename))

    df = pd.read_csv(path, skiprows=1, sep=';')

    if (filename == 'so_co_so_kham_chua_benh_co_so_cap_quan_ly'):
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

    df.to_csv(os.path.join(os.getcwd(), 'Data', 'Yte_VanHoa_TheThao_MucSongDanCu_TratTuAnToanXaHoi_MoiTruong', 'Yte', '{}.tm3'.format(filename)), sep='\t', index=False)

if __name__ == '__main__':
    # danso(filename='Dân số trung bình phân theo địa phương, giới tính và thành thị nông thôn')
    # danso(filename='danso_diaphuong_thanhthi_nongthon')
    # laodong(filename='laodong_15t_cachtinh_nhomtuoi_nam')
    # laodong(filename='laodong_15t_gioitinh_thanhthi_nongthon_cachtinh_nam_phanto')
    # giaoduc(filename='giaoduc_chitieu_nam')
    # congnghiep(filename='congnghiep')
    #dautu(filename='vondautu_thanhphankt_cachtinh_nam')
    #xaydung(filename='dientich_xaydung_loainha_nam')
    #lamnghiep(filename='lamnghiep_dientich_chiso_nam_loai')
    #dulich(filename='ketqua_kinhdoanh_nganhdulich')
    #dulich(filename='doanhthu_theo_gia_thuc_te_thanh_phan_kinh_te')
    #thuongmai(filename='tytrong_giatri_xuatkhau_nhapkhau_tongsanpham_trongnuoc')
    #thuongmai(filename='tri_gia_xuat_khau_hang_hoa_theo_khu_vuc_kinh_te_nhom_hang')
    #thuongmai(filename='tri_gia_nhap_khau_hang_hoa_theo_khu_vuc')
    #trattuantoanxahoi(filename='trattu_antoan_xahoi')
    #thethao(filename='so_huy_chuong_the_thao')
    #vanhoa(filename='xuat_ban_sach_van_hoa_pham_bao_tap_chi')
    yte(filename='so_co_so_kham_chua_benh_co_so_cap_quan_ly')
    pass
    