import numpy as np
import csv
import time

class model:
    def __init__(self, **kargs):
        self.input_count = kargs['input_count']
        self.output_count = kargs['output_count']
        self.rnd_mean = kargs['rnd_mean']
        self.rnd_std = kargs['rnd_std']

    def default_randomize(): np.random.seed(1234)

    def randomize(): np.random.seed(time.time())

    def load_abalone_dataset(self):
        with open('./abalone.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)
            rows = []
            for row in csvreader:
                rows.append(row)

        self.data = np.zeros([len(rows), self.input_count+self.output_count])
        for idx, row in enumerate(rows):
            if row[0] == 'I': self.data[idx, 0] = 1
            if row[0] == 'M': self.data[idx, 1] = 1
            if row[0] == 'F': self.data[idx, 2] = 1
            self.data[idx, 3:] = row[1:]

    def init_model(self):
        self.weight = np.random.normal(self.rnd_mean, self.rnd_std, [self.input_count, self.output_count])
        self.bias = np.zeros([self.output_count])

    def train_and_test(self, epoch_count, mb_size, report):
        step_count = self.arrange_data(mb_size)
        test_x, test_y = self.get_test_data()

        for epoch in range(epoch_count):
            losses, accs = [], []

            for idx in range(step_count):
                train_x, train_y = self.get_train_data(mb_size, idx)
                loss, acc = self.run_train(train_x, train_y)
                losses.append(loss)
                accs.append(acc)

            if report > 0 and (epoch+1) % report == 0:
                acc = self.run_test(test_x, test_y)
                print('Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'.format(epoch+1, np.mean(losses), np.mean(accs), acc))
        
        final_acc = self.run_test(test_x, test_y)
        print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))

    def arrange_data(self, mb_size):
        self.shuffle_map = np.arrange(self.data.shape[0])
        np.random.shuffle(self.shuffle_map)
        step_count = int(self.data.shape[0] * 0.8) // mb_size
        self.test_begin_idx = step_count * mb_size
        return step_count
    
    def get_test_data(self):
        test_data = self.data[self.shuffle_map[self.test_begin_idx:]]
        return test_data[:, :-self.output_count], test_data[:, -self.output_count:]

    def get_train_data(self, mb_size, idx):
        if idx == 0:
            np.random.shuffle(self.shuffle_map[:self.test_begin_idx])
        train_data = self.data[self.shuffle_map[mb_size*idx:mb_size*(idx+1)]]
        return train_data[:, :-self.output_count], train_data[:, -self.output_count:]

    def run_train(self, x, y):
        output, aux_nn = self.forward_neuralnet(x)
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(output, y)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        return loss, accuracy

    def run_test(self, x, y):
        output, _ = self.forward_neuralnet(x)
        accuracy = self.eval_accuracy(output, y)
        return accuracy

    