import wfdb
import util
import ft

def test_warelet_denoising():
    rec_name = f'../data/ecg_data/Person_01/rec_2'
    record = wfdb.rdrecord(rec_name, channels = [0]) 

    data = record.p_signal.flatten()
    rdata1 = ft.median_denoising(data, record.fs)
    rdata2 = ft.wavelet_denoising(rdata1)
    util.show_plot(rdata2)

if __name__ == '__main__':
    test_warelet_denoising()