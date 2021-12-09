import numpy as np
import csv

def load_abalone_dataset(input_count, output_count):
    with open('./abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    data = np.zeros([len(rows), input_count+output_count])
    for idx, row in enumerate(rows):
        if row[0] == 'I': data[idx, 0] = 1
        if row[0] == 'M': data[idx, 1] = 1
        if row[0] == 'F': data[idx, 2] = 1
        data[idx, 3:] = row[1:]

    return data

def load_pulsar_dataset():
    with open('./pulsar_stars.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []

        for row in csvreader: rows.append(row)
        data = np.asarray(rows, dtype='float32')

    return data