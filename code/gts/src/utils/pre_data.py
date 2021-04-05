import random
import json
import copy
import re
import os
import pandas as pd
import nltk
import pdb

PAD_token = 0

class Lang:
	"""
	class to save the vocab and two dict: the word->index and index->word
	"""
	def __init__(self):
		self.word2index = {}
		self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count word tokens
		self.num_start = 0

	def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
		for word in sentence:
			if re.search("N\d+|NUM|\d+", word):
				continue
			if word not in self.index2word:
				self.word2index[word] = self.n_words
				self.word2count[word] = 1
				self.index2word.append(word)
				self.n_words += 1
			else:
				self.word2count[word] += 1

	def trim(self, logger, min_count):  # trim words below a certain count threshold
		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		logger.debug('keep_words {} / {} = {}'.format(len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)))

		# Reinitialize dictionaries
		self.word2index = {}
		# self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count default tokens

		for word in keep_words:
			self.word2index[word] = self.n_words
			self.index2word.append(word)
			self.n_words += 1

	def build_input_lang(self, logger, trim_min_count):  # build the input lang vocab and dict
		if trim_min_count > 0:
			self.trim(logger, trim_min_count)
			self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
		else:
			self.index2word = ["PAD", "NUM"] + self.index2word
		self.word2index = {}
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
						  ["SOS", "UNK"]
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.num_start = len(self.index2word)

		self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
		self.n_words = len(self.index2word)

		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

def load_raw_data(data_path, dataset, is_train = True):  # load the data to list(dict())
	train_ls = None
	if is_train:
		train_path = os.path.join(data_path, dataset, 'train.csv')
		train_df = pd.read_csv(train_path)
		train_ls = train_df.to_dict('records')

	dev_path = os.path.join(data_path, dataset, 'dev.csv')
	dev_df = pd.read_csv(dev_path)
	dev_ls = dev_df.to_dict('records')

	return train_ls, dev_ls

# remove the superfluous brackets
def remove_brackets(x):
	y = x
	if x[0] == "(" and x[-1] == ")":
		x = x[1:-1]
		flag = True
		count = 0
		for s in x:
			if s == ")":
				count -= 1
				if count < 0:
					flag = False
					break
			elif s == "(":
				count += 1
		if flag:
			return x
	return y

def transfer_num(train_ls, dev_ls, chall=False):  # transfer num into "NUM"
	print("Transfer numbers...")
	dev_pairs = []
	generate_nums = []
	generate_nums_dict = {}
	copy_nums = 0

	if train_ls != None:
		train_pairs = []
		for d in train_ls:
			# nums = []
			nums = d['Numbers'].split()
			input_seq = []
			seg = nltk.word_tokenize(d["Question"].strip())
			equation = d["Equation"].split()

			numz = ['0','1','2','3','4','5','6','7','8','9']
			opz = ['+', '-', '*', '/']
			idxs = []
			for s in range(len(seg)):
				if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
					input_seq.append("NUM")
					idxs.append(s)
				else:
					input_seq.append(seg[s])
			if copy_nums < len(nums):
				copy_nums = len(nums)

			out_seq = []
			for e1 in equation:
				if len(e1) >= 7 and e1[:6] == "number":
					out_seq.append('N'+e1[6:])
				elif e1 not in opz:
					generate_nums.append(e1)
					if e1 not in generate_nums_dict:
						generate_nums_dict[e1] = 1
					else:
						generate_nums_dict[e1] += 1
					out_seq.append(e1)
				else:
					out_seq.append(e1)

			train_pairs.append((input_seq, out_seq, nums, idxs))
	else:
		train_pairs = None

	for d in dev_ls:
		# nums = []
		nums = d['Numbers'].split()
		input_seq = []
		seg = nltk.word_tokenize(d["Question"].strip())
		equation = d["Equation"].split()

		numz = ['0','1','2','3','4','5','6','7','8','9']
		opz = ['+', '-', '*', '/']
		idxs = []
		for s in range(len(seg)):
			if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
				input_seq.append("NUM")
				idxs.append(s)
			else:
				input_seq.append(seg[s])
		if copy_nums < len(nums):
			copy_nums = len(nums)

		out_seq = []
		for e1 in equation:
			if len(e1) >= 7 and e1[:6] == "number":
				out_seq.append('N'+e1[6:])
			elif e1 not in opz:
				generate_nums.append(e1)
				if e1 not in generate_nums_dict:
					generate_nums_dict[e1] = 1
				else:
					generate_nums_dict[e1] += 1
				out_seq.append(e1)
			else:
				out_seq.append(e1)
		if chall:
			dev_pairs.append((input_seq, out_seq, nums, idxs, d['Type'], d['Variation Type'], d['Annotator'], d['Alternate']))
		else:
			dev_pairs.append((input_seq, out_seq, nums, idxs))

	temp_g = []
	for g in generate_nums_dict:
		if generate_nums_dict[g] >= 5:
			temp_g.append(g)
	return train_pairs, dev_pairs, temp_g, copy_nums

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
	res = []
	for word in sentence:
		if len(word) == 0:
			continue
		if word in lang.word2index:
			res.append(lang.word2index[word])
		else:
			res.append(lang.word2index["UNK"])
	if "EOS" in lang.index2word and not tree:
		res.append(lang.word2index["EOS"])
	return res

def sentence_from_indexes(lang, indexes):
	sent = []
	for ind in indexes:
		sent.append(lang.index2word[ind])
	return sent

def prepare_data(config, logger, pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, input_lang=None, output_lang=None, tree=False):
	if input_lang == None:
		input_lang = Lang()
	if output_lang == None:
		output_lang = Lang()

	test_pairs = []
	train_pairs = None

	if pairs_trained != None:
		train_pairs = []
		for pair in pairs_trained:
			if not tree:
				input_lang.add_sen_to_vocab(pair[0])
				output_lang.add_sen_to_vocab(pair[1])
			elif pair[-1]:
				input_lang.add_sen_to_vocab(pair[0])
				output_lang.add_sen_to_vocab(pair[1])

	if config.embedding == 'bert' or config.embedding == 'roberta':
		for pair in pairs_tested:
			if not tree:
				input_lang.add_sen_to_vocab(pair[0])
			elif pair[-1]:
				input_lang.add_sen_to_vocab(pair[0])

	if pairs_trained != None:

		input_lang.build_input_lang(logger, trim_min_count)
		if tree:
			output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
		else:
			output_lang.build_output_lang(generate_nums, copy_nums)

		for pair in pairs_trained:
			num_stack = []
			for word in pair[1]: # For each token in equation
				temp_num = []
				flag_not = True
				if word not in output_lang.index2word: # If token is not in output vocab
					flag_not = False
					for i, j in enumerate(pair[2]):
						if j == word:
							temp_num.append(i) # Append number list index of token not in output vocab

				if not flag_not and len(temp_num) != 0: # Equation has an unknown token and it is a number present in number list (could be default number with freq < 5)
					num_stack.append(temp_num)
				if not flag_not and len(temp_num) == 0: # Equation has an unknown token but it is not a number from number list
					num_stack.append([_ for _ in range(len(pair[2]))])

			num_stack.reverse()
			input_cell = indexes_from_sentence(input_lang, pair[0])
			output_cell = indexes_from_sentence(output_lang, pair[1], tree)
			train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
								pair[2], pair[3], num_stack))

	logger.debug('Indexed {} words in input language, {} words in output'.format(input_lang.n_words, output_lang.n_words))

	for pair in pairs_tested:
		num_stack = []
		for word in pair[1]:
			temp_num = []
			flag_not = True
			if word not in output_lang.index2word:
				flag_not = False
				for i, j in enumerate(pair[2]):
					if j == word:
						temp_num.append(i)

			if not flag_not and len(temp_num) != 0:
				num_stack.append(temp_num)
			if not flag_not and len(temp_num) == 0:
				num_stack.append([_ for _ in range(len(pair[2]))])

		num_stack.reverse()
		input_cell = indexes_from_sentence(input_lang, pair[0])
		output_cell = indexes_from_sentence(output_lang, pair[1], tree)
		if config.challenge_disp:
			test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
						   pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], pair[7]))
		else:
			test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
						   pair[2], pair[3], num_stack))

	return input_lang, output_lang, train_pairs, test_pairs

# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
	seq += [PAD_token for _ in range(max_length - seq_len)]
	return seq

# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
	pairs = copy.deepcopy(pairs_to_batch)
	random.shuffle(pairs)  # shuffle the pairs
	pos = 0
	input_lengths = []
	output_lengths = []
	nums_batches = []
	batches = []
	input_batches = []
	output_batches = []
	num_stack_batches = []  # save the num stack which
	num_pos_batches = []
	num_size_batches = []
	while pos + batch_size < len(pairs):
		batches.append(pairs[pos:pos+batch_size])
		pos += batch_size
	batches.append(pairs[pos:])

	for batch in batches:
		batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
		input_length = []
		output_length = []
		for _, i, _, j, _, _, _ in batch:
			input_length.append(i)
			output_length.append(j)
		input_lengths.append(input_length)
		output_lengths.append(output_length)
		input_len_max = input_length[0]
		output_len_max = max(output_length)
		input_batch = []
		output_batch = []
		num_batch = []
		num_stack_batch = []
		num_pos_batch = []
		num_size_batch = []
		for i, li, j, lj, num, num_pos, num_stack in batch:
			num_batch.append(len(num))
			input_batch.append(pad_seq(i, li, input_len_max))
			output_batch.append(pad_seq(j, lj, output_len_max))
			num_stack_batch.append(num_stack)
			num_pos_batch.append(num_pos)
			num_size_batch.append(len(num_pos))
		input_batches.append(input_batch)
		nums_batches.append(num_batch)
		output_batches.append(output_batch)
		num_stack_batches.append(num_stack_batch)
		num_pos_batches.append(num_pos_batch)
		num_size_batches.append(num_size_batch)
	return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches

def get_num_stack(eq, output_lang, num_pos):
	num_stack = []
	for word in eq:
		temp_num = []
		flag_not = True
		if word not in output_lang.index2word:
			flag_not = False
			for i, j in enumerate(num_pos):
				if j == word:
					temp_num.append(i)
		if not flag_not and len(temp_num) != 0:
			num_stack.append(temp_num)
		if not flag_not and len(temp_num) == 0:
			num_stack.append([_ for _ in range(len(num_pos))])
	num_stack.reverse()
	return num_stack