import pickle
from datetime import datetime
from datetime import timedelta
from Corpus import Dataset, DataLoader, DataReader
import numpy as np
from utils import sigmoid, softmax, sigmoid_backward_simple
import os

class Tagger:
    def __init__(self, model_path=None):
        self.model_name = 'BPNN Tagger'
        self.model = None
        self.data_reader = None
        if model_path:
            self.load_model(model_path)

    class Model:
        def __init__(self, input_size, output_size, embedding_size, hidden_size):
            self.input_size = input_size
            self.output_size = output_size
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size
            self.embed = None
            self.data = None
            self.e = None
            self.input = np.random.randn(input_size * embedding_size, hidden_size) / np.sqrt(hidden_size)
            self.input_bias = np.random.randn(hidden_size)
            self.u_3 = None
            self.hidden = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
            self.hidden_bias = np.random.randn(hidden_size)
            self.u_6 = None
            self.output = np.random.randn(hidden_size, output_size) / np.sqrt(output_size)
            self.output_bias = np.random.randn(output_size)
            self.u_9 = None
            self.out = None

        def forward(self, data):
            batch_size, input_size = data.shape  # data shape (batch_size, input_size)
            self.data = data
            data = self.embed[data]  # (batch_size, input_size, embedding_size)
            data = data.reshape((batch_size, -1))  # (batch_size, input_size * embedding_size)
            self.e = data

            data = data @ self.input  # (batch, hidden)
            data = data + self.input_bias
            data = sigmoid(data)
            self.u_3 = data

            data = data @ self.hidden  # (batch, hidden)
            data = data + self.hidden_bias
            data = sigmoid(data)
            self.u_6 = data

            out = data @ self.output  # (batch, output_size)
            out = out + self.output_bias
            # out = sigmoid(out)
            # self.u_9 = out

            out = softmax(out, 1)
            self.out = out
            return out

        def backward(self, ground_truth, lr=0.01):
            # ground_truth -> (batch_size)
            batch_size = ground_truth.shape[0]
            y = np.zeros((batch_size, self.output_size))
            y[np.arange(batch_size), ground_truth] = 1

            u8_grad = self.out - y                              # (batch, out)
            out_grad = self.u_6.T @ u8_grad                     # (hidden, batch) @ (batch, out) -> (hidden, out)
            u6_grad = u8_grad @ self.output.T                   # (batch, out) @ (out, hidden) -> (batch, hidden)

            u5_grad = sigmoid_backward_simple(self.u_6) * u6_grad     # (batch, hidden)
            hidden_grad = self.u_3.T @ u5_grad                  # (hidden, batch) @ (batch, hidden) -> (hidden, hidden)
            u3_grad = u5_grad @ self.hidden.T                   # (batch, hidden) @ (hidden, hidden) -> (batch, hidden)

            u2_grad = sigmoid_backward_simple(self.u_3) * u3_grad     # (batch, hidden)
            input_grad = self.e.T @ u2_grad                     # (i * e, batch) @ (batch, hidden) -> (i * e,h)
            e_grad = u2_grad @ self.input.T                     # (batch, hidden) @ (hidden, i * e) -> (batch, i * e)
            e_grad = e_grad.reshape((batch_size, self.input_size, self.embedding_size))

            self.output_bias -= lr * np.sum(u8_grad, 0)
            self.output = (1 - 0.5 / 437991) * self.output - lr * out_grad
            self.hidden_bias -= lr * np.sum(u5_grad, 0)
            self.hidden = (1 - 0.5 / 437991) * self.hidden - lr * hidden_grad
            self.input_bias -= lr * np.sum(u2_grad, 0)
            self.input = (1 - 0.5 / 437991) * self.input - lr * input_grad

            using_data = list(set(self.data.reshape(-1)))

            for data in using_data:
                self.embed[data] -= lr * np.sum(e_grad[self.data == data], 0)

    class Config:
        def __init__(self, stop_threshold=0,
                     max_iter=30,
                     check_point=None,
                     save_iter=5,
                     hidden_size=100,
                     lr=0.01):
            self.stop_threshold = stop_threshold
            self.max_iter = max_iter
            self.check_point = check_point
            self.save_iter = save_iter
            self.hidden_size = hidden_size
            self.lr = lr

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model, self.data_reader = pickle.load(file)

    def evaluate(self, eval_dataset: Dataset):
        dl = DataLoader(eval_dataset)
        right = 0
        word_count = 0
        for x, y in dl:
            x = np.array(x, dtype='int')
            y = np.array(y, dtype='int')
            tag = np.argmax(self.model.forward(x), -1)
            right += np.sum(y == tag)
            word_count += len(y)
        return right, word_count, right / word_count

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump((self.model, self.data_reader), file)

    def tag(self, s, index=None):
        pass

    def train(self, data_path, test_path=None, dev_path=None, embedding_path=None, config=None):
        if config is None:
            config = self.Config()

        stop_threshold = config.stop_threshold
        max_iter = config.max_iter
        check_point = config.check_point
        save_iter = config.save_iter
        hidden_size = config.hidden_size
        lr = config.lr

        dr = DataReader(data_path, embedding_path)
        self.data_reader = dr
        train_dataset = dr.to_dataset(data_path)
        dev_dataset = dr.to_dataset(dev_path)
        test_dataset = dr.to_dataset(test_path)

        train_loader = DataLoader(train_dataset, config.batch_size, True)
        print(f"Set the seed for built-in generating random numbers to 1")
        np.random.seed(1)
        print(f"Set the seed for numpy generating random numbers to 1")

        self.model = self.Model(5, dr.tag_size, dr.embed_size, hidden_size)
        self.model.embed = dr.embed

        convergence = False
        iter_count = 0
        times = []

        while not convergence:
            start = datetime.now()
            inside_count = 0
            loss = 0
            for x, y in train_loader:
                x = np.array(x, dtype='int')
                y = np.array(y, dtype='int')
                batch_size = len(y)
                out = self.model.forward(x)
                self.model.backward(y, lr)
                loss += -np.average(np.log(out[np.arange(batch_size), y]))
                inside_count += 1
                if inside_count % 8000 == 0:
                    print(f"loss {loss / inside_count}")
                    tag_id = np.argmax(out, -1)
                    tag = [dr.tag_reverse_dict[t_id] for t_id in tag_id]
                    gt = [dr.tag_reverse_dict[t_id] for t_id in y]
                    print(f"tag {tag}")
                    print(f"gt  {gt}")
            iter_count += 1

            _, _, train_acc = self.evaluate(eval_dataset=train_dataset)
            print(f"iter: {iter_count} train accuracy: {train_acc :.5%}")
            _, _, dev_acc = self.evaluate(eval_dataset=dev_dataset)
            print(f"iter: {iter_count} dev   accuracy: {dev_acc :.5%}")
            _, _, test_acc = self.evaluate(eval_dataset=test_dataset)
            print(f"iter: {iter_count} test  accuracy: {test_acc :.5%}")

            end = datetime.now()
            spend = end - start
            times.append(spend)
            loss = loss / inside_count

            if loss <= stop_threshold or iter_count >= max_iter:
                convergence = True
                avg_spend = sum(times, timedelta(0)) / len(times)
                print(f"iter: training average spend time: {avg_spend}s\n")
                if check_point:
                    self.save_model(os.path.join(check_point,'check_point_finish.pickle'))
            else:
                if check_point and (iter_count % save_iter) == 0:
                    self.save_model(os.path.join(check_point, f'check_point_{iter_count}.pickle'))
                print(f"iter: {iter_count} spend time: {spend}s\n")

