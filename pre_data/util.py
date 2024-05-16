import matplotlib.pyplot as plt
import matplotlib
import wfdb

def show_plot2(sign_1, name1, sign_2, name2):
    plt.subplot(2, 1, 1)
    plt.ylabel("mv")
    plt.title(name1)
    plt.plot(sign_1)
    plt.subplot(2, 1, 2)
    plt.title(name2)
    plt.plot(sign_2)
    plt.show()
    
def show3_plot(sign_1, sign_2, sign_3):
    plt.subplot(3, 1, 1)
    plt.ylabel("mv")
    plt.title("原始心电信号")
    plt.plot(sign_1)

    plt.subplot(3, 1, 2)
    plt.ylabel("mv")
    plt.title("基线")
    plt.plot(sign_2)

    plt.subplot(3, 1, 3)
    plt.ylabel("mv")
    plt.title("中值滤波结果")
    plt.plot(sign_3)
    plt.show()
    # plt.savefig('/mnt/c/Users/Gigalo/Desktop/消噪效果.png')

def show_plot(sign, name):
    plt.subplot(1, 1, 1)
    plt.ylabel("mv")
    plt.title(name)
    plt.plot(sign)
    plt.show()

def get_test_data():
    rec_name = f'../data/ecg_data/Person_01/rec_1'
    record = wfdb.rdrecord(rec_name, channels = [0]) 
    return record 