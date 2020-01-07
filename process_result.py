import sys
import os

def parse_result(result_path):
    dev = [0]
    train = [0]
    test = [0]
    have_test = False
    max_iter = 0
    final_iter = 0
    time = None
    with open(result_path, 'r') as res_file:
        lines = res_file.readlines()
        pointer = 0
        while True:
            if pointer < len(lines):
                line = lines[pointer]
            else:
                break
            if not line.startswith('iter'):
                pointer += 1
            else:
                split_line = line.split()
                now_iter = int(split_line[1])
                train_acc = float(split_line[-1][:-1])
                train.append(train_acc)
                pointer += 1
                dev_acc = float(lines[pointer].split()[-1][:-1])
                if dev_acc > max(dev):
                    max_iter = now_iter
                dev.append(dev_acc)
                pointer += 1
                split_line = lines[pointer].split()
                if split_line[2] == 'test':
                    test_acc = float(split_line[-1][:-1])
                    test.append(test_acc)
                    have_test = True
                    pointer += 1
                    split_line = lines[pointer].split()
                    if split_line[2] == 'average':
                        time = split_line[-1]
                        final_iter = now_iter
                        break
                    else:
                        pointer += 1
                elif split_line[2] == 'average':
                    time = split_line[-1]
                    final_iter = now_iter
                    break
                else:
                    pointer += 1

    return max_iter, final_iter, train[max_iter], dev[max_iter], test[max_iter] if have_test else '-', time


if __name__ == '__main__':
    path = './result'
    print(f'|  name  | 特征优化 | 大数据 | 权重叠加 | 随机学习率 | 模拟退火 |  iter  |  train  |  dev  |  test  |  time  |')
    print('| :----------------------: | :---: | :---: | :---: | :---: | :---: | :------: | :------: | :------: | :------: | :------------: |')
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_name = os.path.splitext(file)
        ext = file_name[1]
        if ext == '.res':
            names = file_name[0].split('_')
            attr = names[1:]
            split_name = names[0].split()
            if split_name[0] == 'Optimized':
                name = ' '.join(split_name[1:])
                Optimized = '√'
            else:
                name = names[0]
                Optimized = '×'
            big = '√' if 'big' in attr else '×'
            ap = '√' if 'perceptron' in attr else '×'
            random = '√' if 'random' in attr else '×'
            anneal = '√' if 'anneal' in attr else '×'
            max_iter, final_iter, train, dev, test, time = parse_result(file_path)
            print(f'|  {name}  | {Optimized} | {big} | {ap} | {random} | {anneal} |  {max_iter}/{final_iter}  |  {train}  |  {dev}  |  {test}  |  {time}  |')