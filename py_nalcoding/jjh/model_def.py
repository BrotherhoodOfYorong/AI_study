import numpy as np
import time
import data_loader

class model:
    def __init__(self, **kargs):
        self.input_count = kargs['input_count']
        self.output_count = kargs['output_count']
        self.rnd_mean = kargs['rnd_mean']
        self.rnd_std = kargs['rnd_std']
        self.m_type = kargs['m_type']
        self.adjust = False if not 'adjust' in kargs.keys() else kargs['adjust']
        self.hidden_config = kargs['hidden_config'] if 'hidden_config' in kargs.keys() else False

    def default_randomize(self): np.random.seed(1234)

    def randomize(self): np.random.seed(int(time.time()))

    def load_dataset(self, chapter):
        if chapter == 1: self.data = data_loader.load_abalone_dataset(self.input_count, self.output_count)
        if chapter == 2: self.data = data_loader.load_pulsar_dataset(adjust=self.adjust)
        if chapter == 3: self.data = data_loader.load_steel_dataset()

    def init_model(self):
        if self.hidden_config != False:
            self.pm_hiddens = []
            prev_count = self.input_count

            for hidden_count in self.hidden_config:
                self.pm_hiddens.append(self.alloc_param_pair([prev_count, hidden_count]))
                prev_count = hidden_count
            
            self.pm_output = self.alloc_param_pair([prev_count, self.output_count])
        else:
            params = self.alloc_param_pair([self.input_count, self.output_count])
            self.weight = params['w']
            self.bias = params['b']

    def alloc_param_pair(self, shape):
        weight = np.random.normal(self.rnd_mean, self.rnd_std, shape)
        bias = np.zeros(shape[-1])
        return {'w':weight, 'b':bias}

    def train_and_test(self, epoch_count, mb_size, report, learning_rate):
        step_count = self.arrange_data(mb_size)
        test_x, test_y = self.get_test_data()
        self.learning_rate = learning_rate

        for epoch in range(epoch_count):
            losses, accs = [], []

            for idx in range(step_count):
                train_x, train_y = self.get_train_data(mb_size, idx)
                loss, acc = self.run_train(train_x, train_y)
                losses.append(loss)
                accs.append(acc)

            if report > 0 and (epoch+1) % report == 0:
                acc = self.run_test(test_x, test_y)
                acc_str = ', '.join(['%5.3f']*len(acc))%tuple(acc) if self.m_type=='binary decision' else '{:5.3f} / {:5.3f}'.format(np.mean(accs), acc)
                print('[Epoch {:02d}] loss: {:5.3f} | results: {}'.format(epoch+1, np.mean(losses), acc_str))
        
        final_acc = self.run_test(test_x, test_y)
        final_acc_str = ', '.join(['%5.3f']*len(final_acc))%tuple(final_acc) if self.m_type=='binary decision' else '%5.3f'%final_acc
        print('\n[Final Test] final results: {}'.format(final_acc_str))

    def arrange_data(self, mb_size):
        self.shuffle_map = np.arange(self.data.shape[0])
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
        self.backprop_neuralnet(G_output, aux_nn, self.learning_rate)

        return loss, accuracy

    def run_test(self, x, y):
        output, _ = self.forward_neuralnet(x)
        accuracy = self.eval_accuracy(output, y)
        return accuracy

    def forward_neuralnet(self, x):
        if self.hidden_config != False:
            hidden = x
            hiddens = [x]
            for pm_hidden in self.pm_hiddens:
                hidden = self.relu(np.matmul(hidden, pm_hidden['w']) + pm_hidden['b'])
                hiddens.append(hidden)
            output = np.matmul(hidden, self.pm_output['w']) + self.pm_output['b']
            return output, hiddens
        else:
            output = np.matmul(x, self.weight) + self.bias
            return output, x

    def backprop_neuralnet(self, G_output, aux, learning_rate):
        if self.hidden_config != False:
            hiddens = aux

            g_output_w_out = hiddens[-1].transpose()
            G_w_out = np.matmul(g_output_w_out, G_output)
            G_b_out = np.sum(G_output, axis=0)

            g_output_hidden = self.pm_output['w'].transpose()
            G_hidden = np.matmul(G_output, g_output_hidden)
            self.pm_output['w'] -= learning_rate * G_w_out
            self.pm_output['b'] -= learning_rate * G_b_out

            for n in reversed(range(len(self.pm_hiddens))):
                G_hidden = G_hidden * self.relu_derv(hiddens[n+1])
                g_hidden_w_hid = hiddens[n].transpose()
                G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
                G_b_hid = np.sum(G_hidden, axis=0)

                g_hidden_hidden = self.pm_hiddens[n]['w'].transpose()
                G_hidden = np.matmul(G_hidden, g_hidden_hidden)

                self.pm_hiddens[n]['w'] -= learning_rate * G_w_hid
                self.pm_hiddens[n]['b'] -= learning_rate * G_b_hid
        else:
            g_output_w = aux.transpose()

            G_w = np.matmul(g_output_w, G_output)
            G_b = np.sum(G_output, axis=0)

            self.weight -= learning_rate * G_w
            self.bias -= learning_rate * G_b

    def forward_postproc(self, output, y):
        if self.m_type=='regression':
            diff = output - y
            square = np.square(diff)
            loss = np.mean(square)
            return loss, diff
        elif self.m_type=='binary decision' or self.m_type=='classification':
            if self.m_type=='binary decision'   : loss_function = self.sigmoid_corss_entropy_with_logits
            if self.m_type=='classification'    : loss_function = self.softmax_cross_entropy_with_logits
            entropy = loss_function(y, output)
            loss = np.mean(entropy)
            return loss, [y, output, entropy]

    def backprop_postproc(self, G_loss, diff):
        if self.m_type=='regression':
            shape = diff.shape

            g_loss_square = np.ones(shape) / np.prod(shape)
            g_square_diff = 2 * diff
            g_diff_output = 1

            G_square = g_loss_square * G_loss
            G_diff = g_square_diff * G_square
            G_output = g_diff_output * G_diff
            return G_output
        elif self.m_type=='binary decision' or self.m_type=='classification':
            y, output, entropy = diff

            g_loss_entropy = 1.0 / np.prod(entropy.shape)
            if self.m_type=='binary decision'   : loss_function = self.sigmoid_cross_entropy_with_logits_derv
            if self.m_type=='classification'    : loss_function = self.softmax_cross_entropy_with_logits_derv

            g_entropy_output = loss_function(y, output)
            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy
            return G_output

    def eval_accuracy(self, output, y):
        if self.m_type == 'regression':
            mdiff = np.mean(np.abs((output-y)/y))
            return 1 - mdiff
        elif self.m_type == 'binary decision':
            est_yes = np.greater(output, 0)
            ans_yes = np.greater(y, 0.5)
            est_no = np.logical_not(est_yes)
            ans_no = np.logical_not(ans_yes)

            tp = np.sum(np.logical_and(est_yes, ans_yes))
            fp = np.sum(np.logical_and(est_yes, ans_no))
            fn = np.sum(np.logical_and(est_no, ans_yes))
            tn = np.sum(np.logical_and(est_no, ans_no)) # ?????? fn tn ????????? ????????????!

            accuracy = self.safe_div(tp+tn, tp+tn+fp+fn)
            precision = self.safe_div(tp, tp+fp)
            recall = self.safe_div(tp, tp+fn) # ????????? fn, tn ????????? ????????? ???????????????
            f1 = 2 * self.safe_div(precision*recall, precision+recall)
            return [accuracy, precision, recall, f1]
        elif self.m_type == 'classification':
            estimate = np.argmax(output, axis=1)
            answer = np.argmax(y, axis=1)
            correct = np.equal(estimate, answer)
            return np.mean(correct)

    ############### Mathematical Functions ###############
    def relu(self, x):
        return np.maximum(x, 0)

    def relu_derv(self, y):
        return np.sign(y)

    def sigmoid(self, x):
        return np.exp(-self.relu(-x)) / (1.0 + np.exp(-np.abs(x)))

    def sigmoid_derv(self, x, y):
        return y * (1 - y)

    def sigmoid_corss_entropy_with_logits(self, z, x):
        return self.relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

    def sigmoid_cross_entropy_with_logits_derv(self, z, x):
        return -z + self.sigmoid(x)

    def safe_div(self, p, q):
        p, q = float(p), float(q)
        if np.abs(q) < 1.0e-20: return np.sign(p)
        return p / q

    def softmax(self, x):
        max_elemental = np.max(x, axis=1)
        diff = (x.transpose() - max_elemental).transpose()
        exp = np.exp(diff)
        sum_exp = np.sum(exp, axis=1)
        probabilities = (exp.transpose()/sum_exp).transpose()
        return probabilities
    
    def softmax_derv(self, x, y):
        mb_size, nom_size = x.shape
        derv = np.ndarray([mb_size, nom_size, nom_size])
        for n in range(mb_size):
            for i in range(nom_size):
                for j in range(nom_size):
                    derv[n, i, j] = -y[n, i] * y[n, y]
                derv[n, i, i] += y[n, i]
        return derv

    def softmax_cross_entropy_with_logits(self, labels, logits):
        probabilities = self.softmax(logits)
        return -np.sum(labels * np.log(probabilities+1.0e-10), axis=1)

    def softmax_cross_entropy_with_logits_derv(self, labels, logits):
        return self.softmax(logits) - labels