# write a script that reads csv data and averages each column

import csv
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    """ Reads a CSV file and returns the data as a numpy array. """
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return np.array(data, dtype=float)

def average_columns(data):
    """ Averages each column in a list of lists. """
    num_columns = len(data[0])
    averages = [0] * num_columns

    for row in data:
        for i, value in enumerate(row):
            averages[i] += float(value)

    averages = [avg / len(data) for avg in averages]
    return averages

def smooth(y, window_size=10):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

if __name__ == "__main__":
    file_path = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\hollow_steel1.csv"
    file_path2 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\hollow_steel2.csv"
    file_path3 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\hollow_steel3.csv"
    file_path4 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\rubber3_1.csv"
    file_path5 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\rubber3_2.csv"
    file_path6 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\rubber3_3.csv"
    data = read_csv(file_path)
    data2 = read_csv(file_path2)
    data3 = read_csv(file_path3)
    data4 = read_csv(file_path4)
    data5 = read_csv(file_path5)
    data6 = read_csv(file_path6)
    # average rows 206 to 235 (inclusive)
    data_avg = data4[-1]
    print(data_avg)
    # averages = average_columns(data_avg)
    q = data_avg[1]  # average heat flux
    r = 0.043  # distance between the flux sensor and the thermocouple
    Th = data_avg[2]  # average temperature of the flux sensor
    Tc = data_avg[4]  # average temperature of the thermocouple
    print(q * r / (Th - Tc))

    # # plot the data
    # plt.plot(data[30:300 - 9, 0] - data[30, 0], smooth(data[30:300, 2]), color='blue')
    # plt.plot(data2[20:300 - 9, 0] - data2[20, 0], smooth(data2[20:300, 2]), color='blue')
    # plt.plot(data3[50:300 - 9, 0] - data3[50, 0], smooth(data3[50:300, 2]), color='blue')
    # plt.plot(data4[0:300 - 9, 0] - data4[15, 0], smooth(data4[0:300, 2]), color='red')
    # plt.plot(data5[15:300 - 9, 0] - data5[15, 0], smooth(data5[15:300, 2]), color='red')
    # plt.plot(data6[0:300 - 9, 0], smooth(data6[0:300, 2]), color='red')

    # plt.xlabel("Time (s)")
    # plt.ylabel("Temperature (C)")
    # plt.title("Temperature vs. Time")
    # plt.legend(["Steel1", "Steel2", "Steel3", "Rubber1", "Rubber2", "Rubber3"])
    # plt.show()