import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import os
import sys
from src.utils.bleu import compute_bleu
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU

		Args:
			gpu_num (int): Which GPU to use
		Returns:
			device (torch.device): GPU device
	'''

	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device

def create_save_directories(path):
	if not os.path.exists(path):
		os.makedirs(path)

def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			state (dict): model state
			epoch (int): current epoch
			logger (logger): logger variable to log messages
			model_path (string): directory to save models
			ckpt (string): checkpoint name
	'''

	ckpt_path = os.path.join(model_path, '{}_{}.pt'.format(ckpt, epoch))
	logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)

def get_latest_checkpoint(model_path, logger):
	'''
		Looks for the checkpoint with highest epoch number in the directory "model_path" 

		Args:
			model_path (string): directory where model is saved
			logger (logger): logger variable to log messages
		Returns:
			ckpt_path: checkpoint path to the latest checkpoint 
	'''

	ckpts = glob('{}/*.pt'.format(model_path))
	ckpts = sorted(ckpts)

	if len(ckpts) == 0:
		logger.warning('No Checkpoints Found')

		return None
	else:
		latest_epoch = max([int(x.split('_')[-1].split('.')[0]) for x in ckpts])
		ckpts = sorted(ckpts, key= lambda x: int(x.split('_')[-1].split('.')[0]) , reverse=True )
		ckpt_path = ckpts[0]
		logger.info('Checkpoint found with epoch number : {}'.format(latest_epoch))
		logger.debug('Checkpoint found at : {}'.format(ckpt_path))

		return ckpt_path

def load_checkpoint(model, mode, ckpt_path, logger, device):
	'''
		Load the model at checkpoint

		Args:
			model (object of class TransformerModel): model
			mode (string): train or test mode
			ckpt_path: checkpoint path to the latest checkpoint 
			logger (logger): logger variable to log messages
			device (torch.device): GPU device
		Returns:
			start_epoch (int): epoch from which to start
			min_train_loss (float): minimum train loss
			min_val_loss (float): minimum validation loss
			max_train_acc (float): maximum train accuracy
			max_val_acc (float): maximum validation accuracy score
			max_val_bleu (float): maximum valiadtion bleu score
			best_epoch (int): epoch with highest validation accuracy
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
	'''

	checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	start_epoch = checkpoint['epoch']
	min_train_loss  =checkpoint['min_train_loss']
	min_val_loss = checkpoint['min_val_loss']
	voc1 = checkpoint['voc1']
	voc2 = checkpoint['voc2']
	max_train_acc = checkpoint['max_train_acc']
	max_val_acc = checkpoint['max_val_acc']
	max_val_bleu = checkpoint['max_val_bleu']
	best_epoch = checkpoint['best_epoch']

	model.to(device)

	if mode == 'train':
		model.train()
	else:
		model.eval()

	logger.info('Successfully Loaded Checkpoint from {}, with epoch number: {} for {}'.format(ckpt_path, start_epoch, mode))

	return start_epoch, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2

class Voc1:
	def __init__(self):
		self.trimmed = False
		self.frequented = False
		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		self.w2c = {}
		self.nwords = 3

	def add_word(self, word):
		if word not in self.w2id:
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		else:
			self.w2c[word] += 1

	def add_sent(self, sent):
		for word in sent.split():
			self.add_word(word)

	def most_frequent(self, topk):
		# if self.frequented == True:
		# 	return
		# self.frequented = True

		keep_words = []
		count = 3
		sort_by_value = sorted(
			self.w2c.items(), key=lambda kv: kv[1], reverse=True)
		for word, freq in sort_by_value:
			keep_words += [word]*freq
			count += 1
			if count == topk:
				break

		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		self.w2c = {}
		self.nwords = 3

		for word in keep_words:
			self.add_word(word)

	def trim(self, mincount):
		if self.trimmed == True:
			return
		self.trimmed = True

		keep_words = []
		for k, v in self.w2c.items():
			if v >= mincount:
				keep_words += [k]*v

		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		self.w2c = {}
		self.nwords = 3
		for word in keep_words:
			self.addWord(word)

	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, args, train_dataloader):
		for data in train_dataloader:
			for sent in data['ques']:
				self.add_sent(sent)

		self.most_frequent(args.vocab_size)
		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords

	def add_to_vocab_dict(self, args, dataloader):
		for data in dataloader:
			for sent in data['ques']:
				self.add_sent(sent)

		self.most_frequent(args.vocab_size)
		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords

class Voc2:
	def __init__(self, config):
		self.frequented = False
		if config.mawps_vocab:
			# '0.25', '8.0', '0.05', '60.0', '7.0', '5.0', '2.0', '4.0', '1.0', '12.0', '100.0', '25.0', '0.1', '3.0', '0.01', '0.5', '10.0'
			self.w2id = {'<s>': 11, '</s>': 1, '+': 2, '-': 3, '*': 4, '/': 5, 'number0': 6, 'number1': 7, 'number2': 8, 'number3': 9, 'number4': 10, '0.25': 0, '8.0': 12, '0.05': 13, '60.0': 14, '7.0': 15, '5.0': 16, '2.0': 17, '4.0': 18, '1.0': 19, '12.0': 20, '100.0': 21, '25.0': 22, '0.1': 23, '3.0': 24, '0.01': 25, '0.5': 26, '10.0': 27, 'unk': 28}
			self.id2w = {11: '<s>', 1: '</s>', 2: '+', 3: '-', 4: '*', 5: '/', 6: 'number0', 7: 'number1', 8: 'number2', 9: 'number3', 10: 'number4', 0: '0.25', 12: '8.0', 13: '0.05', 14: '60.0', 15: '7.0', 16: '5.0', 17: '2.0', 18: '4.0', 19: '1.0', 20: '12.0', 21: '100.0', 22: '25.0', 23: '0.1', 24: '3.0', 25: '0.01', 26: '0.5', 27: '10.0', 28: 'unk'}
			self.w2c = {'+': 0, '-': 0, '*': 0, '/': 0, 'number0': 0, 'number1': 0, 'number2': 0, 'number3': 0, 'number4': 0, '0.25': 0, '8.0': 0, '0.05': 0, '60.0': 0, '7.0': 0, '5.0': 0, '2.0': 0, '4.0': 0, '1.0': 0, '12.0': 0, '100.0': 0, '25.0': 0, '0.1': 0, '3.0': 0, '0.01': 0, '0.5': 0, '10.0': 0, 'unk': 0}
			self.nwords = 29
		else:
			self.w2id = {'<s>': 11, '</s>': 1, '+': 2, '-': 3, '*': 4, '/': 5, 'number0': 6, 'number1': 7, 'number2': 8, 'number3': 9, 'number4': 10, 'number5': 0, 'unk': 12}
			self.id2w = {11: '<s>', 1: '</s>', 2: '+', 3: '-', 4: '*', 5: '/', 6: 'number0', 7: 'number1', 8: 'number2', 9: 'number3', 10: 'number4', 0: 'number5', 12: 'unk'}
			self.w2c = {'+': 0, '-': 0, '*': 0, '/': 0, 'number0': 0, 'number1': 0, 'number2': 0, 'number3': 0, 'number4': 0, 'number5': 0, 'unk': 0}
			self.nwords = 13 # For some reason, model outputs NANs if I keep <s> as token 0 - That's because it was masking out the <s> token in make_len_mask in model

	def add_word(self, word):
		if word not in self.w2id: # IT SHOULD NEVER GO HERE!!
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		else:
			self.w2c[word] += 1

	def add_sent(self, sent):
		for word in sent.split():
			self.add_word(word)

	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, args, train_dataloader):
		for data in train_dataloader:
			for sent in data['eqn']:
				self.add_sent(sent)

		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords

	def add_to_vocab_dict(self, args, dataloader):
		for data in dataloader:
			for sent in data['eqn']:
				self.add_sent(sent)

		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords
 
def bleu_scorer(ref, hyp, script='default'):
	'''
		Bleu Scorer (Send list of list of references, and a list of hypothesis)
	'''
	
	refsend = []
	for i in range(len(ref)):
		refsi = []
		for j in range(len(ref[i])):
			refsi.append(ref[i][j].split())
		refsend.append(refsi)

	gensend = []
	for i in range(len(hyp)):
		gensend.append(hyp[i].split())

	if script == 'nltk':
		 metrics = corpus_bleu(refsend, gensend)
		 return [metrics]

	metrics = compute_bleu(refsend, gensend)
	return metrics