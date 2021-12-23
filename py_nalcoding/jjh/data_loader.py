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

def load_pulsar_dataset(adjust=False):
    pulsars, stars = [], []
    with open('./pulsar_stars.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)

        for row in csvreader:
            if row[8] == '1': pulsars.append(row)
            else: stars.append(row)
    star_cnt, pulsar_cnt = len(stars), len(pulsars)

    if adjust:
        data = np.zeros([2*star_cnt, 9])
        data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
        for n in range(star_cnt):
            data[star_cnt+n] = np.asarray(pulsars[n%pulsar_cnt], dtype='float32')
    else:
        data = np.zeros([star_cnt+pulsar_cnt, 9])
        data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
        data[star_cnt:, :] = np.asarray(pulsars, dtype='float32')

    return data

def load_steel_dataset():
    with open('./faults.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)

        rows = []
        for row in csvreader:
            rows.append(row)
    
    return np.asarray(rows, dtype='float32')