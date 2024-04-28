import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation

import os

def recurrence_plot(data, threshold=0.1):
    # Calculate the distance matrix
    N = len(data)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = np.abs(data[i] - data[j])

    # Create the recurrence plot
    recurrence_plot = np.where(distance_matrix <= threshold, 1, 0)
    return recurrence_plot


def test_rp(result):
    transformer = RecurrencePlot(dimension=2, time_delay=2)
    X_rp = transformer.transform(result)
    plt.figure(1)
    plt.imshow(X_rp[0], cmap='rainbow', origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'rp_picture/person_{1111}_{11111}.png', bbox_inches='tight')


def rp(result, id):
    z = 1
    for m in range(2, 5):
        for t in range(1, 11):
            transformer = RecurrencePlot(dimension=m, time_delay=t)
            X_rp = transformer.transform(result)
            save_path = f'rp_picture/person_{id}_{z}.png'
            os.makedirs(os.path.dirname(save_path))
            plt.figure(1)
            plt.imshow(X_rp[0], cmap='rainbow', origin='lower')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_path, bbox_inches='tight')
            z += 1


def gadf(nums, result, id, is_test=False):
    # PAA降维
    pid = 1
    for L in nums:
        L = int(L)
        transformer = PiecewiseAggregateApproximation(
            window_size=None, 
            output_size=L
        )
        resultPAA = transformer.transform(result)
        N1 = np.array([resultPAA[1, :]])
        gadf = GramianAngularField(method='difference')
        X_gadf = gadf.fit_transform(N1)
        # get the save path 
        save_path = f'../image/test_image/person{id}/{pid}.png' if is_test else f'../image/train_image/person{id}/{pid}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')
        print(f'has loaded picture of person_{id}_{pid}')
        pid += 1


def test():
    time_points = np.linspace(1, 10000, 10000)  # type array
    print(type(time_points))
    print(len(time_points))


if __name__ == '__main__':
    test()
