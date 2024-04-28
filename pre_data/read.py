#!/usr/bin/python3
import wfdb
import numpy as np
import scipy.io as scio
from scipy.io import loadmat

import ft
import ts


def read_ecg_data(person_id, rec_id):
    # 读取原始数据记录
    rec_name = f'../data/ecg_data/Person_{person_id}/rec_{rec_id}'
    record = wfdb.rdrecord(rec_name, channels=[0])
    data = record.p_signal.flatten()
    # 中值滤波
    rdata1 = ft.median_denoising(data, record.fs)
    # 小波消噪处理
    rdata2 = ft.wavelet_denoising(rdata1)
    return rdata2


def gen_mat():
    # 生成对应矩阵
    matrix = np.mat([read_ecg_data(str(i).rjust(2, '0'), '1')
                    for i in range(1, 91)])
    scio.savemat('ecg.mat', {'data': matrix})


def gen_rp_pic():
    ecg_mat = loadmat("ecg.mat")['data']
    time_points = np.linspace(1, 1000, 1000)
    time_points = time_points.tolist()

    for i in range(0, 1):
        FHR1 = (ecg_mat[i, 0:1000])
        result = [FHR1]
        ts.rp(result, i + 1)


def test_rrp():
    ecg_mat = loadmat("ecg.mat")['data']
    time_points = np.linspace(1, 1000, 1000)                  # type array
    time_points = time_points.tolist()                  # 转list
    FHR1 = (ecg_mat[0, 0:1000])
    result = [FHR1]
    ts.test_rp(result)


def gen_gadf_pic(n):
    ecg_mat = loadmat("ecg.mat")['data']
    # 等差取参，维度[180,360,...,7200]
    nums = np.linspace(180, 7200, 40)
    time_points = np.linspace(1, 7200, 7200)
    time_points = time_points.tolist()

    # futures = []
    # pool = multiprocessing.Pool(processes=1)
    for i in range(40, n):
        FHR1 = (ecg_mat[i, 0:7200]).tolist()
        result = [time_points, FHR1]
        ts.gadf(nums, result, i + 1)
        # pool.apply_async(ts.gadf, (nums, result, i + 1))
    # pool.close()
    # pool.join()

def gen_gadf_test_pic(n):
    ecg_mat = loadmat("ecg.mat")['data']
    nums = np.linspace(180, 7200, 40)
    time_points = np.linspace(1, 7200, 7200)
    time_points = time_points.tolist()

    for i in range(85, n):
        FHR1 = (ecg_mat[i, 2800:]).tolist()
        result = [time_points, FHR1]
        ts.gadf(nums, result, i + 1, is_test=True)

def test_read():
    gen_mat()
    dict = scio.loadmat('ecg.mat')
    data = dict['data']
    print(np.shape(data))


def test_read_attr():
    rec_name = '../data/ecg_data/Person_01/rec_1'
    annotation = wfdb.rdann(rec_name, 'atr')
    print(annotation.sample)
    print(annotation.symbol)

def main():
    # gen_mat()
    # gen_gadf_pic(40)
    # gen_gadf_pic(90)
    gen_gadf_test_pic(90)
    # test_read()


if __name__ == '__main__':
    main()
