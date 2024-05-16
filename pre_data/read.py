#!/usr/bin/python3
import wfdb
from wfdb import processing
import numpy as np
import scipy.io as scio
from scipy.io import loadmat
import multiprocessing
import ft
import ts
import neurokit2 as nk
import math

# 一些参数
person_ct = 90
pic_ct = 20
all_data_len = 10000
pic_len = 10000 // pic_ct

def read_ecg_data(person_id, rec_id, qrs_size):
    # 读取原始数据记录
    rec_name = f'../ecg_data/Person_{person_id}/rec_{rec_id}'
    try:
        record = wfdb.rdrecord(rec_name, channels=[1])
    except Exception as e:
        return np.empty((0, ), dtype=float, order = 'C')
    data = record.p_signal.flatten()
    # 中值滤波
    rdata1 = ft.median_denoising(data, record.fs)
    # rdata2 = ft.wavelet_denoising(rdata1)

    _, results = nk.ecg_peaks(rdata1, sampling_rate=record.fs)
    rpeaks = results["ECG_R_Peaks"]
    _, waves_peak = nk.ecg_delineate(rdata1, rpeaks, sampling_rate=record.fs, method="peak")
    p_peaks = waves_peak['ECG_P_Peaks']
    rdata3 = np.empty((0, ), dtype = float, order = 'C')
    for pos in p_peaks[1:qrs_size + 1]:
        if math.isnan(pos):
            continue
        wdata = rdata1[pos : pos + pic_len]
        if len(wdata) == pic_len:
            rdata3 = np.append(rdata3, wdata)
    return rdata3

def read_all_data_row(person_id, rec_id):
    rec_name = f'../ecg_data/Person_{person_id}/rec_{rec_id}'
    record = wfdb.rdrecord(rec_name, channels=[1])
    data = record.p_signal.flatten()
    rdata1 = ft.median_denoising(data, record.fs)
    return rdata1

def read_mat_row(person_id, is_test=False):
    res = np.empty([0, ], dtype = float, order = 'C')
    if is_test:
        res = np.append(res, read_ecg_data(person_id, 2, pic_ct))
    else:
        res = np.append(res, read_ecg_data(person_id, 1, pic_ct))
    print(person_id, res.size)
    return res

def repair_row(person_id, row, num):
    row = np.append(row, read_ecg_data(person_id, 2, num))
    return row

def gen():
    matrix = [read_all_data_row(str(i).rjust(2, '0'), 1)
                    for i in range(1, person_ct + 1)]
    matrix = np.mat(matrix)
    scio.savemat('ecg.mat', {'data': matrix})
    matrix2 = [read_all_data_row(str(i).rjust(2, '0'), 2)
                    for i in range(1, person_ct + 1)]
    matrix2 = np.mat(matrix2)
    scio.savemat('ecg_test.mat', {'data': matrix2})

def gen_mat():
    # 生成对应矩阵
    matrix = [read_mat_row(str(i).rjust(2, '0'))
                    for i in range(1, person_ct + 1)]
    for idx, row in enumerate(matrix):
        while matrix[idx].shape != (all_data_len, ):
            print(idx, matrix[idx].shape)
            num =  (all_data_len - len(matrix[idx])) // pic_len
            print(idx, matrix[idx].shape, num)
            matrix[idx] = repair_row(str(idx + 1).rjust(2, '0'), matrix[idx], num)
    matrix = np.mat(matrix)
    scio.savemat('ecg.mat', {'data': matrix})

def gen_test_mat():
    # 生成对应矩阵
    matrix = [read_mat_row(str(i).rjust(2, '0'), is_test=True)
                    for i in range(1, person_ct + 1)]
    for idx, row in enumerate(matrix):
        while matrix[idx].shape != (all_data_len, ):
            print(idx, matrix[idx].shape)
            num =  (all_data_len - len(matrix[idx])) // pic_len
            matrix[idx] = repair_row(str(idx + 1).rjust(2, '0'), matrix[idx], num)
    matrix = np.mat(matrix)
    scio.savemat('ecg_test.mat', {'data': matrix})

def gen_gadf_pic(n):
    ecg_mat = loadmat("ecg.mat")['data']
    # 等差取参，维度[180,360,...,7200]
    nums = np.linspace(pic_len, all_data_len, pic_ct)
    # time_points = np.linspace(1, all_data_len, all_data_len)
    # time_points = time_points.tolist()

    # futures = []
    pool = multiprocessing.Pool(processes=4)
    for i in range(0, n):
        FHR1 = ecg_mat[i, :]
        # result = [time_points, FHR1]
        # ts.singal_gadf(nums, FHR1, 350, i + 1)
        pool.apply_async(ts.singal_gadf, (nums, FHR1, pic_len, i + 1))
    pool.close()
    pool.join()

def gen_gadf_test_pic(n):
    ecg_mat = loadmat("ecg_test.mat")['data']
    nums = np.linspace(pic_len, all_data_len, pic_ct)
    # time_points = np.linspace(1, 7000, 7000)
    # time_points = time_points.tolist()

    pool = multiprocessing.Pool(processes=4)
    for i in range(0, n):
        # rdata = read_mat_row(str(i + 1).rjust(2, '0'), is_test=True)
        FHR1 = ecg_mat[i, :]
        # result = [time_points, FHR1]
        # ts.gadf(nums, result, i + 1, is_test=True)
        pool.apply_async(ts.singal_gadf, (nums, FHR1, pic_len, i + 1, True))
    pool.close()
    pool.join()

# other test function
def test_read():
    # gen_mat()
    dict = scio.loadmat('ecg_test.mat')
    data = dict['data']
    print(np.shape(data))

def test_read_attr():
    rec_name = '../ecg_data/Person_01/rec_1'
    annotation = wfdb.rdann(rec_name, 'atr')
    print(annotation.sample)
    print(annotation.symbol)

def main():
    gen()
    # gen_mat()
    # gen_test_mat()
    # test_read()
    gen_gadf_pic(90)
    gen_gadf_test_pic(90)

if __name__ == '__main__':
    main()
