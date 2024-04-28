from scipy.io import loadmat                    # 科学计算库，eg:差值运算、优化、图像处理、数学统计
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler

FHRdata = loadmat("ecg.mat")['data']
# 不需要分段聚合近似,PAA
# N1 = np.array([FHRdata])           # 转array (1,10000),作为GAF的输入

# PAA降维
FHRN1 = FHRdata[0, 0:7440]                 # type ndarray
'''原理分步可视化'''
FHRN11 = FHRN1.tolist()                   # 转list
time_points = np.linspace(1, 7440, 7440)  # type array
time_points1 = time_points.tolist()       # 转list
result = [time_points1, FHRN11]           # type list
result = np.array(result)                 # 转array (2,7440), 作为GAF的输入

# 0.分段聚合近似,PAA取参
transformer = PiecewiseAggregateApproximation(window_size=10)
result = transformer.transform(result)

# 1.缩放至[-1,1]
scaler = MinMaxScaler()
scaled_X = scaler.transform(result)
plt.figure(1)
plt.plot(scaled_X[0, :], scaled_X[1, :])
plt.title('After scaling')
plt.xlabel('timestamp')
plt.ylabel('FHR[bmp]')
plt.show()

# 2.转换至极坐标
arccos_X = np.arccos(scaled_X[1, :])
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(result[0, :], arccos_X)
ax.grid(True)
ax.set_title('Polar coordinates', va='bottom')
ax.set_rmax(1.6)
ax.set_rticks([0.4, 0.8, 1.2, 1.6])  # 减少径向刻度
ax.set_rlabel_position(-22.5)    # 将径向标签从绘图线移开
plt.show()

# 3.GASF / GADF转换
plt.figure(3)
field = [a+b for a in arccos_X for b in arccos_X]
gram = np.cos(field).reshape(-1, 744)     # 744=总长/window_size
plt.imshow(gram, cmap='rainbow', origin='lower')
plt.show()