import os
import sys
from collections import Counter

word_count_threshold = 25
training_label_data_path = sys.argv[1]
training_label_index_data_path = './training_label_index.txt'
nolabel_data_path = sys.argv[2]
nolabel_index_data_path = './training_nolabel_index.txt'
dict_path = './dict.txt'

def prepocessing_data(label_data_path, nolabel_data_path, testing_data_path, word_count_threshold):
    c = Counter()
    with open(label_data_path, 'r') as train_data:
        for line in train_data:
            label, text = line.strip().split(' +++$+++ ')
            words = text.split(' ')
            for word in words:
                c[word] += 1
    with open(nolabel_data_path, 'r') as train_data:
        for line in train_data:
            words = line.strip().split(' ')
            for word in words:
                c[word] += 1
    '''
    with open(testing_data_path, 'r') as test_data:
        test_data.readline()
        for line in test_data:
            first_comma = line.index(',')
            text = line.strip()[first_comma+1:]
            words = text.split(' ')
            for word in words:
                c[word] += 1
    '''
    c = Counter({k: c for k, c in c.items() if c >= word_count_threshold}) 
    d = write_dict(dict_path, c)
    return None
    
def write_dict(dict_path, c):
    d = ['unk']
    if os.path.isfile(dict_path):
        os.remove(dict_path)
    with open(dict_path, 'a') as dict_txt:
        dict_txt.write('unk\r\n')
        for key, cnt in list(c.items()):
            dict_txt.write(key +'\r\n')
            d.append(key)

def read_dict(dict_path):
    d = []
    with open(dict_path, 'r') as dict_txt:
        for line in dict_txt:
            d.append(line.strip())
    return d

def indexing_training_data(label_data_path, nolabel_data_path, label_index_data_path, nolabel_index_data_path, dict_path):
    d = read_dict(dict_path)
    with open(label_data_path, 'r') as train_data:
        all_index_text = []
        all_label = []
        counter = 0
        for line in train_data:
            counter += 1
            if (counter % 20000) == 0:
                print(counter)
            label, text = line.strip().split(' +++$+++ ')
            all_label.append(label)
            words = text.split(' ')
            index_text = ''
            for word in words:
                if word in d:
                    index_text += str(d.index(word) + 1) +' '
                else:
                    index_text += str(1) +' ' 
            all_index_text.append(index_text.strip())
        if os.path.isfile(label_index_data_path):
            os.remove(label_index_data_path)
        with open(label_index_data_path, 'a') as index_data:
            for cntr in range(len(all_index_text)):
                index_data.write(all_label[cntr] +','+ all_index_text[cntr] +'\r\n')
    with open(nolabel_data_path, 'r') as train_data:
        all_index_text = []
        counter = 0
        for line in train_data:
            counter += 1
            if (counter % 20000) == 0:
                print(counter)
            words = line.strip().split(' ')
            index_text = ''
            for word in words:
                if word in d:
                    index_text += str(d.index(word) + 1) +' '
                else:
                    index_text += str(1) +' ' 
            all_index_text.append(index_text.strip())
        if os.path.isfile(nolabel_index_data_path):
            os.remove(nolabel_index_data_path)
        with open(nolabel_index_data_path, 'a') as index_data:
            for index_text in all_index_text:
                index_data.write(index_text +'\r\n')
    return None

def indexing_testing_data(testing_data_path, testing_data_index_path, dict_path):
    d = read_dict(dict_path)
    with open(testing_data_path, 'r') as test_data:
        all_index_text = []
        test_data.readline()
        counter = 0
        for line in test_data:
            counter += 1
            if (counter % 20000) == 0:
                print(counter)
            first_comma = line.index(',')
            index = line.strip()[:first_comma]
            text = line.strip()[first_comma+1:]
            index_text = ''
            words = text.split(' ')
            for word in words:
                if word in d:
                    index_text += str(d.index(word) + 1) +' '
                else:
                    index_text += str(1) +' ' 
            all_index_text.append(index_text.strip())
        if os.path.isfile(testing_data_index_path):
            os.remove(testing_data_index_path)
        with open(testing_data_index_path, 'a') as index_data:
            index = 0
            for index_text in all_index_text:
                index_data.write(str(index) +','+ index_text +'\r\n')
                index += 1
    return None

def main():

	prepocessing_data(training_label_data_path, nolabel_data_path, '', word_count_threshold)
	indexing_training_data(training_label_data_path, nolabel_data_path, 
                       training_label_index_data_path, nolabel_index_data_path, dict_path)

if __name__ == '__main__':
	main()