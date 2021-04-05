import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import os
import sys
# from src.utils.bleu import compute_bleu
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from src.utils.pre_data import *

def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU
	'''
	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device

def create_save_directories(path):
	if not os.path.exists(path):
		os.makedirs(path)

def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op

def index_batch_to_words(input_batch, input_length, lang):
	'''
		Args:
			input_batch: List of BS x Max_len
			input_length: List of BS
		Return:
			contextual_input: List of BS
	'''
	contextual_input = []
	for i in range(len(input_batch)):
		contextual_input.append(stack_to_string(sentence_from_indexes(lang, input_batch[i][:input_length[i]])))

	return contextual_input

def sort_by_len(seqs, input_len, device=None, dim=1):
	orig_idx = list(range(seqs.size(dim)))
	# pdb.set_trace()

	# Index by which sorting needs to be done
	sorted_idx = sorted(orig_idx, key=lambda k: input_len[k], reverse=True)
	sorted_idx= torch.LongTensor(sorted_idx)
	if device:
		sorted_idx = sorted_idx.to(device)

	sorted_seqs = seqs.index_select(1, sorted_idx)
	sorted_lens=  [input_len[i] for i in sorted_idx]

	# For restoring original order
	orig_idx = sorted(orig_idx, key=lambda k: sorted_idx[k])
	orig_idx = torch.LongTensor(orig_idx)
	if device:
		orig_idx = orig_idx.to(device)
		# sorted_lens = torch.LongTensor(sorted_lens).to(device)
	return sorted_seqs, sorted_lens, orig_idx

def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			model state
			epoch number
			logger variable
			directory to save models
			checkpoint name
	'''
	ckpt_path = os.path.join(model_path, '{}.pt'.format(ckpt))
	logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)

def load_checkpoint(config, embedding, encoder, predict, generate, merge, mode, ckpt_path, logger, device,
					embedding_optimizer = None, encoder_optimizer = None, predict_optimizer = None, generate_optimizer = None, merge_optimizer = None,
					embedding_scheduler = None, encoder_scheduler = None, predict_scheduler = None, generate_scheduler = None, merge_scheduler = None
					):
	checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

	embedding.load_state_dict(checkpoint['embedding_state_dict'])
	encoder.load_state_dict(checkpoint['encoder_state_dict'])
	predict.load_state_dict(checkpoint['predict_state_dict'])
	generate.load_state_dict(checkpoint['generate_state_dict'])
	merge.load_state_dict(checkpoint['merge_state_dict'])

	if mode == 'train':
		embedding_optimizer.load_state_dict(checkpoint['embedding_optimizer_state_dict'])
		encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
		predict_optimizer.load_state_dict(checkpoint['predict_optimizer_state_dict'])
		generate_optimizer.load_state_dict(checkpoint['generate_optimizer_state_dict'])
		merge_optimizer.load_state_dict(checkpoint['merge_optimizer_state_dict'])

		embedding_scheduler.load_state_dict(checkpoint['embedding_scheduler_state_dict'])
		encoder_scheduler.load_state_dict(checkpoint['encoder_scheduler_state_dict'])
		predict_scheduler.load_state_dict(checkpoint['predict_scheduler_state_dict'])
		generate_scheduler.load_state_dict(checkpoint['generate_scheduler_state_dict'])
		merge_scheduler.load_state_dict(checkpoint['merge_scheduler_state_dict'])

	start_epoch = checkpoint['epoch']
	min_train_loss  = checkpoint['min_train_loss']
	max_train_acc = checkpoint['max_train_acc']
	max_val_acc = checkpoint['max_val_acc']
	equation_acc = checkpoint['equation_acc']
	best_epoch = checkpoint['best_epoch']
	generate_nums = checkpoint['generate_nums']

	embedding.to(device)
	encoder.to(device)
	predict.to(device)
	generate.to(device)
	merge.to(device)

	logger.info('Successfully Loaded Checkpoint from {}, with epoch number: {} for {}'.format(ckpt_path, start_epoch, mode))

	if mode == 'train':
		embedding.train()
		encoder.train()
		predict.train()
		generate.train()
		merge.train()
	else:
		embedding.eval()
		encoder.eval()
		predict.eval()
		generate.eval()
		merge.eval()		

	return start_epoch, min_train_loss, max_train_acc, max_val_acc, equation_acc, best_epoch, generate_nums

def get_latest_checkpoint(model_path, logger):
	'''
		Looks for the checkpoint with highest epoch number in the directory "model_path" 

		Args:
			model_path: including the run_name
			logger variable: to log messages
		Returns:
			checkpoint: path to the latest checkpoint 
	'''

	ckpts = glob('{}/*.pt'.format(model_path))
	ckpts = sorted(ckpts)

	if len(ckpts) == 0:
		logger.warning('No Checkpoints Found')

		return None
	else:
		#pdb.set_trace()
		#latest_epoch = max([int(x.split('_')[-1].split('.')[0]) for x in ckpts])
		#ckpts = sorted(ckpts, key= lambda x: int(x.split('_')[-1].split('.')[0]) , reverse=True )
		ckpt_path = ckpts[0]
		#logger.info('Checkpoint found with epoch number : {}'.format(latest_epoch))
		logger.debug('Checkpoint found at : {}'.format(ckpt_path))

		return ckpt_path