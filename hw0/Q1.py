import sys
from collections import Counter

def word_count(file_path):

	cntr = Counter()
	term_list = []
	with open(file_path, 'r') as fopen:
		for word in fopen.readline().split(' '):
			term = word.strip()
			cntr[term] += 1
			if term not in term_list:
				term_list.append(term)

	return term_list, cntr

def write_answer(term_list, cntr):
	
	with open('Q1.txt', 'w') as fwrite:
		for i in range(len(term_list)):
			fwrite.write('%s %d %d' % (term_list[i], i, cntr[term_list[i]]))
			if i != len(term_list) - 1:
				fwrite.write('\n')

if __name__ == '__main__':

	wc = word_count(sys.argv[1])
	write_answer(wc[0], wc[1])