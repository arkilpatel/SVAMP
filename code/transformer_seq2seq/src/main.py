# python -m src.main -no-debug -mode train -gpu 1 -dropout 0.1 -heads 4 -encoder_layers 1 -decoder_layers 1 -d_model 768 -d_ff 256 -lr 0.0001 -emb_lr 1e-5 -batch_size 32 -epochs 70 -embedding roberta -emb_name roberta-base -mawps_vocab -dataset mawps_fold0 -run_name mawps_try1
import os
import sys
import math
import logging
import pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log, store_results, store_val_results
from src.dataloader import TextDataset
from src.model import build_model, train_model, run_validation

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'

def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Load Datasets'''
		train_set = TextDataset(data_path=data_path, dataset=config.dataset,
								datatype='train', max_length=config.max_length, is_debug=config.debug, is_train=True)
		val_set = TextDataset(data_path=data_path, dataset=config.dataset,
							  datatype='dev', max_length=config.max_length, is_debug=config.debug, grade_info=config.grade_disp, 
							  type_info=config.type_disp, challenge_info=config.challenge_disp)

		'''In case of sort by length, write a different case with shuffle=False '''
		train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		train_size = len(train_dataloader) * config.batch_size
		val_size = len(val_dataloader)* config.batch_size
		
		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
		logger.info(msg)

		return train_dataloader, val_dataloader

	elif config.mode == 'test':
		logger.debug('Loading Test Data...')

		test_set = TextDataset(data_path=data_path, dataset=config.dataset,
							   datatype='test', max_length=config.max_length, is_debug=config.debug)
		test_dataloader = DataLoader(
			test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		logger.info('Test Data Loaded...')
		return test_dataloader

	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))


def main():
	'''read arguments'''
	parser = build_parser()
	args = parser.parse_args()
	config = args
	mode = config.mode
	if mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	if config.full_cv:
		global data_path 
		data_name = config.dataset
		data_path = data_path + data_name + '/'
		config.val_result_path = os.path.join(result_folder, 'CV_results_{}.json'.format(data_name))
		fold_acc_score = 0.0
		folds_scores = []
		for z in range(5):
			run_name = config.run_name + '_fold' + str(z)
			config.dataset = 'fold' + str(z)
			config.log_path = os.path.join(log_folder, run_name)
			config.model_path = os.path.join(model_folder, run_name)
			config.board_path = os.path.join(board_path, run_name)
			config.outputs_path = os.path.join(outputs_folder, run_name)

			vocab1_path = os.path.join(config.model_path, 'vocab1.p')
			vocab2_path = os.path.join(config.model_path, 'vocab2.p')
			config_file = os.path.join(config.model_path, 'config.p')
			log_file = os.path.join(config.log_path, 'log.txt')

			if config.results:
				config.result_path = os.path.join(result_folder, 'val_results_{}_{}.json'.format(data_name, config.dataset))

			if is_train:
				create_save_directories(config.log_path)
				create_save_directories(config.model_path)
				create_save_directories(config.outputs_path)
			else:
				create_save_directories(config.log_path)
				create_save_directories(config.result_path)

			logger = get_logger(run_name, log_file, logging.DEBUG)
			writer = SummaryWriter(config.board_path)

			logger.debug('Created Relevant Directories')
			logger.info('Experiment Name: {}'.format(config.run_name))

			'''Read Files and create/load Vocab'''
			if is_train:
				train_dataloader, val_dataloader = load_data(config, logger)

				logger.debug('Creating Vocab...')

				voc1 = Voc1()
				voc1.create_vocab_dict(config, train_dataloader)

				# Removed
				# voc1.add_to_vocab_dict(config, val_dataloader)

				voc2 = Voc2(config)
				voc2.create_vocab_dict(config, train_dataloader)

				# Removed
				# voc2.add_to_vocab_dict(config, val_dataloader)

				logger.info('Vocab Created with number of words : {}'.format(voc1.nwords))

				with open(vocab1_path, 'wb') as f:
					pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
				with open(vocab2_path, 'wb') as f:
					pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

				logger.info('Vocab saved at {}'.format(vocab1_path))

			else:
				test_dataloader = load_data(config, logger)
				logger.info('Loading Vocab File...')

				with open(vocab1_path, 'rb') as f:
					voc1 = pickle.load(f)
				with open(vocab2_path, 'rb') as f:
					voc2 = pickle.load(f)

				logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

			# TO DO : Load Existing Checkpoints here
			checkpoint = get_latest_checkpoint(config.model_path, logger)

			if is_train:
				model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

				logger.info('Initialized Model')

				if checkpoint == None:
					min_val_loss = torch.tensor(float('inf')).item()
					min_train_loss = torch.tensor(float('inf')).item()
					max_val_bleu = 0.0
					max_val_acc = 0.0
					max_train_acc = 0.0
					best_epoch = 0
					epoch_offset = 0
				else:
					epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																		load_checkpoint(model, config.mode, checkpoint, logger, device)

				with open(config_file, 'wb') as f:
					pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

				logger.debug('Config File Saved')

				logger.info('Starting Training Procedure')
				max_val_acc = train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, 
							epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)

			else:
				gpu = config.gpu
				mode = config.mode
				dataset = config.dataset
				batch_size = config.batch_size
				with open(config_file, 'rb') as f:
					config = AttrDict(pickle.load(f))
					config.gpu = gpu
					config.mode = mode
					config.dataset = dataset
					config.batch_size = batch_size

				with open(config_file, 'rb') as f:
					config = AttrDict(pickle.load(f))
					config.gpu = gpu

				model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

				epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																		load_checkpoint(model, config.mode, checkpoint, logger, device)

				logger.info('Prediction from')
				od = OrderedDict()
				od['epoch'] = ep_offset
				od['min_train_loss'] = min_train_loss
				od['min_val_loss'] = min_val_loss
				od['max_train_acc'] = max_train_acc
				od['max_val_acc'] = max_val_acc
				od['max_val_bleu'] = max_val_bleu
				od['best_epoch'] = best_epoch
				print_log(logger, od)

				test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
				logger.info('Accuracy: {}'.format(test_acc_epoch))

			fold_acc_score += max_val_acc
			folds_scores.append(max_val_acc)

		fold_acc_score = fold_acc_score/5
		store_val_results(config, fold_acc_score, folds_scores)
		logger.info('Final Val score: {}'.format(fold_acc_score))
			
	else:
		'''Run Config files/paths'''
		run_name = config.run_name
		config.log_path = os.path.join(log_folder, run_name)
		config.model_path = os.path.join(model_folder, run_name)
		config.board_path = os.path.join(board_path, run_name)
		config.outputs_path = os.path.join(outputs_folder, run_name)

		vocab1_path = os.path.join(config.model_path, 'vocab1.p')
		vocab2_path = os.path.join(config.model_path, 'vocab2.p')
		config_file = os.path.join(config.model_path, 'config.p')
		log_file = os.path.join(config.log_path, 'log.txt')

		if config.results:
			config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

		if is_train:
			create_save_directories(config.log_path)
			create_save_directories(config.model_path)
			create_save_directories(config.outputs_path)
		else:
			create_save_directories(config.log_path)
			create_save_directories(config.result_path)

		logger = get_logger(run_name, log_file, logging.DEBUG)
		writer = SummaryWriter(config.board_path)

		logger.debug('Created Relevant Directories')
		logger.info('Experiment Name: {}'.format(config.run_name))

		'''Read Files and create/load Vocab'''
		if is_train:
			train_dataloader, val_dataloader = load_data(config, logger)

			logger.debug('Creating Vocab...')

			voc1 = Voc1()
			voc1.create_vocab_dict(config, train_dataloader)

			# Removed
			# voc1.add_to_vocab_dict(config, val_dataloader)

			voc2 = Voc2(config)
			voc2.create_vocab_dict(config, train_dataloader)

			# Removed
			# voc2.add_to_vocab_dict(config, val_dataloader)

			logger.info('Vocab Created with number of words : {}'.format(voc1.nwords))

			with open(vocab1_path, 'wb') as f:
				pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
			with open(vocab2_path, 'wb') as f:
				pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.info('Vocab saved at {}'.format(vocab1_path))

		else:
			test_dataloader = load_data(config, logger)
			logger.info('Loading Vocab File...')

			with open(vocab1_path, 'rb') as f:
				voc1 = pickle.load(f)
			with open(vocab2_path, 'rb') as f:
				voc2 = pickle.load(f)

			logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

		# Load Existing Checkpoints here
		checkpoint = get_latest_checkpoint(config.model_path, logger)

		if is_train:
			model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

			logger.info('Initialized Model')

			if checkpoint == None:
				min_val_loss = torch.tensor(float('inf')).item()
				min_train_loss = torch.tensor(float('inf')).item()
				max_val_bleu = 0.0
				max_val_acc = 0.0
				max_train_acc = 0.0
				best_epoch = 0
				epoch_offset = 0
			else:
				epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																	load_checkpoint(model, config.mode, checkpoint, logger, device)

			with open(config_file, 'wb') as f:
				pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.debug('Config File Saved')

			logger.info('Starting Training Procedure')
			train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, 
						epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)

		else:
			gpu = config.gpu
			mode = config.mode
			dataset = config.dataset
			batch_size = config.batch_size
			with open(config_file, 'rb') as f:
				config = AttrDict(pickle.load(f))
				config.gpu = gpu
				config.mode = mode
				config.dataset = dataset
				config.batch_size = batch_size

			with open(config_file, 'rb') as f:
				config = AttrDict(pickle.load(f))
				config.gpu = gpu

			model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

			epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																	load_checkpoint(model, config.mode, checkpoint, logger, device)

			logger.info('Prediction from')
			od = OrderedDict()
			od['epoch'] = ep_offset
			od['min_train_loss'] = min_train_loss
			od['min_val_loss'] = min_val_loss
			od['max_train_acc'] = max_train_acc
			od['max_val_acc'] = max_val_acc
			od['max_val_bleu'] = max_val_bleu
			od['best_epoch'] = best_epoch
			print_log(logger, od)

			test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
			logger.info('Accuracy: {}'.format(test_acc_epoch))


if __name__ == '__main__':
	main()


''' Just docstring format '''
# class Vehicles(object):
# 	'''
# 	The Vehicle object contains a lot of vehicles

# 	Args:
# 		arg (str): The arg is used for...
# 		*args: The variable arguments are used for...
# 		**kwargs: The keyword arguments are used for...

# 	Attributes:
# 		arg (str): This is where we store arg,
# 	'''
# 	def __init__(self, arg, *args, **kwargs):
# 		self.arg = arg

# 	def cars(self, distance,destination):
# 		'''We can't travel distance in vehicles without fuels, so here is the fuels

# 		Args:
# 			distance (int): The amount of distance traveled
# 			destination (bool): Should the fuels refilled to cover the distance?

# 		Raises:
# 			RuntimeError: Out of fuel

# 		Returns:
# 			cars: A car mileage
# 		'''
# 		pass