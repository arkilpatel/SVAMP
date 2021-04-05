import os
import logging
import pdb
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import unicodedata
from collections import OrderedDict

class TextDataset(Dataset):
	'''
		Expecting csv files with columns ['sent1', 'sent2']

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing
						is_train: 

	'''

	def __init__(self, data_path='./data/', dataset='mawps', datatype='train', max_length=30, is_debug=False, is_train=False, grade_info=False, type_info=False, challenge_info=False):
		if datatype=='train':
			file_path = os.path.join(data_path, dataset, 'train.csv')
		elif datatype=='dev':
			file_path = os.path.join(data_path, dataset, 'dev.csv')
		else:
			file_path = os.path.join(data_path, dataset, 'dev.csv')

		if grade_info:
			self.grade_info = True
		else:
			self.grade_info = False

		if type_info:
			self.type_info = True
		else:
			self.type_info = False

		if challenge_info:
			self.challenge_info = True
		else:
			self.challenge_info = False

		file_df= pd.read_csv(file_path)

		self.ques= file_df['Question'].values
		self.eqn= file_df['Equation'].values
		self.nums= file_df['Numbers'].values
		self.ans= file_df['Answer'].values

		if grade_info:
			self.grade = file_df['Grade'].values

		if type_info:
			self.type = file_df['Type'].values

		if challenge_info:
			self.type = file_df['Type'].values
			self.var_type = file_df['Variation Type'].values
			self.annotator = file_df['Annotator'].values
			self.alternate = file_df['Alternate'].values

		if is_debug:
			self.ques= self.ques[:5000:500]
			self.eqn= self.eqn[:5000:500]

		self.max_length= max_length

		if grade_info and type_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.grade, self.type)
		elif grade_info and not type_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.grade)
		elif type_info and not grade_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.type)
		elif challenge_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.type, self.var_type, self.annotator, self.alternate)
		else:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans)

		if is_train:
			all_sents = sorted(all_sents, key = lambda x : len(x[0].split()))

		if grade_info and type_info:
			self.ques, self.eqn, self.nums, self.ans, self.grade, self.type = zip(*all_sents)
		elif grade_info and not type_info:
			self.ques, self.eqn, self.nums, self.ans, self.grade = zip(*all_sents)
		elif type_info and not grade_info:
			self.ques, self.eqn, self.nums, self.ans, self.type = zip(*all_sents)
		elif challenge_info:
			self.ques, self.eqn, self.nums, self.ans, self.type, self.var_type, self.annotator, self.alternate = zip(*all_sents)
		else:
			self.ques, self.eqn, self.nums, self.ans = zip(*all_sents)

	def __len__(self):
		return len(self.ques)

	def __getitem__(self, idx):
		ques = self.process_string(str(self.ques[idx]))
		eqn = self.process_string(str(self.eqn[idx]))
		nums = self.nums[idx]
		ans = self.ans[idx]

		if self.grade_info and self.type_info:
			grade = self.grade[idx]
			type1 = self.type[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'grade': grade, 
					'type': type1}
		elif self.grade_info and not self.type_info:
			grade = self.grade[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'grade': grade}
		elif self.type_info and not self.grade_info:
			type1 = self.type[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'type': type1}
		elif self.challenge_info:
			type1 = self.type[idx]
			var_type = self.var_type[idx]
			annotator = self.annotator[idx]
			alternate = self.alternate[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'type': type1, 
					'var_type': var_type, 'annotator': annotator, 'alternate': alternate}
	
		return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans}

	def curb_to_length(self, string):
		return ' '.join(string.strip().split()[:self.max_length])

	def process_string(self, string):
		#string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " 's", string)
		string = re.sub(r"\'ve", " 've", string)
		string = re.sub(r"n\'t", " n't", string)
		string = re.sub(r"\'re", " 're", string)
		string = re.sub(r"\'d", " 'd", string)
		string = re.sub(r"\'ll", " 'll", string)
		#string = re.sub(r",", " , ", string)
		#string = re.sub(r"!", " ! ", string)
		#string = re.sub(r"\(", " ( ", string)
		#string = re.sub(r"\)", " ) ", string)
		#string = re.sub(r"\?", " ? ", string)
		#string = re.sub(r"\s{2,}", " ", string)
		return string

