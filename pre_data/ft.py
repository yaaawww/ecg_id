import pywt
import numpy as np
import util
from scipy.signal import medfilt

def median_denoising(ecg_sign, fs):
    f = int(0.8*fs)
    f = f if f % 2 == 1 else f + 1
    # give_up_size = int(f / 2)
    origin_ecg = ecg_sign
    ecg_baseline = medfilt(origin_ecg, f)
    totality_bias = np.sum(ecg_baseline[:])/(len(origin_ecg))
    filtered_ecg = origin_ecg - ecg_baseline
    final_filtered_ecg = filtered_ecg[:] - totality_bias
    return final_filtered_ecg

def wavelet_denoising(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # for cf in coeffs:
        # print(f'{min(cf)}~{max(cf)}')
    # 阈值去噪
    threshold = (np.median(np.abs(cD2)) / 0.6745) * (np.sqrt(2 * np.log(len(cD2))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

if __name__ == '__main__':
    rd = util.get_test_data() 
    origin_ecg = rd.p_signal.flatten()[0:7200]
    f = int(0.8 * rd.fs)
    f = f if f % 2 == 1 else f + 1
    # give_up_size = int(f / 2)
    ecg_baseline = medfilt(origin_ecg, f)
    totality_bias = np.sum(ecg_baseline[:])/(len(origin_ecg))
    filtered_ecg = origin_ecg - ecg_baseline
    final_filtered_ecg = filtered_ecg[:] - totality_bias
    util.show3_plot(origin_ecg, ecg_baseline, final_filtered_ecg)