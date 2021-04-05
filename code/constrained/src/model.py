import os
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.optimization import BertAdam
# from tensorboardX import SummaryWriter
from gensim import models
from src.components.encoder import Encoder
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.bert_encoder import BertEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint, bleu_scorer
from src.utils.evaluate import cal_score, stack_to_string, get_infix_eq
from src.confidence_estimation import *
from collections import OrderedDict



class Seq2SeqModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, num_iters, EOS_tag='</s>', SOS_tag='<s>'):
		super(Seq2SeqModel, self).__init__()

		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.EOS_token = voc2.get_id(EOS_tag)
		self.SOS_token = voc2.get_id(SOS_tag)
		self.logger = logger
		self.num_iters = num_iters

		# To Do: Embeddings from pretrained models like BERT
		# if self.config.use_word2vec:
		# 	config.emb1_size = 300
		# 	self.embedding1  = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze=not self.config.train_word2vec)
		# else:
		# 	self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.emb1_size)
		# 	nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)
		# # self.embedding = nn.Embedding(self.voc.nwords, self.config.emb_size)

		self.embedding2  = nn.Embedding(self.voc2.nwords, self.config.emb2_size)
		nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)

		self.bert = BertEncoder(config.bert_name, self.device)
		# To Do: Vary initialization methods

		self.logger.debug('Building Encoders...')
		self.encoder = Encoder(
			self.config.hidden_size,
			self.config.bert_size,
			self.config.cell_type,
			self.config.depth,
			self.config.dropout,
			self.config.bidirectional
		)
		

		self.logger.debug('Encoders Built...')

		if self.config.use_attn:
			self.decoder    = LuongAttnDecoderRNN(self.config.attn_type,
												  self.embedding2,
												  self.config.cell_type,
												  self.config.hidden_size,
												  self.voc2.nwords,
												  self.config.depth,
												  self.config.dropout).to(device)
		else:
			self.decoder    = DecoderRNN(self.embedding2,
										 self.config.cell_type,
										 self.config.hidden_size,
										 self.voc2.nwords,
										 self.config.depth,
										 self.config.dropout).to(device)



		self.logger.debug('Decoder RNN Built...')


		self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 
		self.criterion = nn.NLLLoss() 

		self.logger.info('All Model Components Initialized...')

	def _form_embeddings(self, file_path):
		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.voc1.nwords, self.config.emb1_size)
		for key, value in self.voc1.id2w.items():
			if value in weights_all:
				weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req	

	def _initialize_optimizer(self):
		self.params = list(self.encoder.parameters()) + \
			list(self.decoder.parameters())

		param_optimizer = list(self.bert.get_model().named_parameters())
		no_decay = ['bias', 'gamma', 'beta']

		optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
		]

		self.bertoptimizer = BertAdam(optimizer_grouped_parameters,
						 lr=self.config.bert_lr,
						 warmup=self.config.warmup,
						 t_total=self.num_iters*self.config.epochs) 

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(self.params, lr=self.config.lr)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
		else:
			self.optimizer = optim.SGD(self.params, lr=self.config.lr)

	# def encode(self, input_seqs, input_len, encoder):
	# 	pdb.set_trace()
	# 	output, hidden = encoder(input_seqs, input_len)
	# 	return output

	# def clf(self, hidden):
	# 	return self.soft_clf(hidden)

	def forward(self, input_seq1, input_seq2, input_len1, input_len2):
		'''
			Args:
				input_seq1 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len1 (tensor): Length of each sequence in input_len1 | size : [batch_size]
				input_seq2 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len2 (tensor): Length of each sequence in input_len2 | size : [batch_size]
			Returns:
				out (tensor) : Probabilities of each output label for each point | size : [batch_size x num_labels]
		'''


	def trainer(self, ques, input_seq2, input_len2, config, device=None ,logger=None):

		self.optimizer.zero_grad()
		self.bertoptimizer.zero_grad()

		input_seq1, input_len1 = self.bert(ques)
		input_seq1 = input_seq1.transpose(0,1)

		encoder_outputs, encoder_hidden = self.encoder(input_seq1, input_len1, self.device)
		
		self.loss =0

		decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device = self.device)

		if config.cell_type == 'lstm':
			decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
		else:
			decoder_hidden = encoder_hidden[:self.decoder.nlayers]

		use_teacher_forcing = True if random.random() < self.config.teacher_forcing_ratio else False
		target_len = max(input_len2)

		if use_teacher_forcing:
			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				self.loss += self.criterion(decoder_output, input_seq2[step])
				decoder_input = input_seq2[step]
		else:
			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				
				topv, topi = decoder_output.topk(1)
				self.loss += self.criterion(decoder_output, input_seq2[step])
				decoder_input = topi.squeeze().detach() 


		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
		self.optimizer.step()
		self.bertoptimizer.step()

		return self.loss.item()/target_len

	# def evaluator(self, input_seq1, input_seq2, input_len1, input_len2, config, device=None ,logger=None):
		



	# 	return acc/batch_size, loss/batch_size

	def greedy_decode(self, ques, input_seq2=None, input_len2=None, validation=False, return_probs = True):
		with torch.no_grad():
			#pdb.set_trace()
			input_seq1, input_len1 = self.bert(ques)
			input_seq1 = input_seq1.transpose(0,1)

			encoder_outputs, encoder_hidden = self.encoder(input_seq1, input_len1, self.device)

			loss =0.0
			decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device=self.device)

			if self.config.cell_type == 'lstm':
				decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
			else:
				decoder_hidden = encoder_hidden[:self.decoder.nlayers]

			decoded_words = [[] for i in range(input_seq1.size(1))]
			decoded_probs = [[] for i in range(input_seq1.size(1))]
			decoder_attentions = []

			if validation:
				target_len = max(input_len2)
			else:
				target_len = self.config.max_length

			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
					decoder_attentions.append(decoder_attention)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				

				if validation:
					loss += self.criterion(decoder_output, input_seq2[step])
				topv, topi = decoder_output.topk(1)
				for i in range(input_seq1.size(1)):
					if topi[i].item() == self.EOS_token:
						continue
					decoded_words[i].append(self.voc2.get_word(topi[i].item()))
					decoded_probs[i].append(topv[i].item())
				decoder_input = topi.squeeze().detach()

			if validation:
				if self.config.use_attn:
					return loss/target_len, decoded_words, decoder_attentions[:step + 1]
				else:
					return loss/target_len, decoded_words, None
			else:
				if return_probs:
					return decoded_words, decoded_probs

				return decoded_words

	def obtain_hidden(self, config, ques, input_seq2=None, input_len2=None):
		with torch.no_grad():
			#pdb.set_trace()
			input_seq1, input_len1 = self.bert(ques)
			input_seq1 = input_seq1.transpose(0,1)

			encoder_outputs, encoder_hidden = self.encoder(input_seq1, input_len1, self.device)

			loss =0.0
			decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device=self.device)


			if self.config.cell_type == 'lstm':
				decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
			else:
				decoder_hidden = encoder_hidden[:self.decoder.nlayers]

			decoded_words = [[] for i in range(input_seq1.size(1))]
			decoder_attentions = []

			hiddens = []

			target_len = max(input_len2)


			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
					decoder_attentions.append(decoder_attention)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

				topv, topi = decoder_output.topk(1)
				for i in range(input_seq1.size(1)):
					if topi[i].item() == self.EOS_token:
						continue
					decoded_words[i].append(self.voc2.get_word(topi[i].item()))
					hiddens.append([self.voc2.get_word(topi[i].item()), hidden[i]])
				decoder_input = topi.squeeze().detach()

			return hiddens, decoded_words

def build_model(config, voc1, voc2, device, logger, num_iters):
	'''
		Add Docstring
	'''
	model = Seq2SeqModel(config, voc1, voc2, device, logger, num_iters)
	model = model.to(device)

	return model



def train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, epoch_offset= 0, min_val_loss=float('inf'), max_val_score=0.0, max_acc_score = 0.0, writer= None):
	'''
		Add Docstring
	'''

	if config.histogram and writer:
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch_offset)
	
	estop_count=0
	
	for epoch in range(1, config.epochs + 1):
		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		print_log(logger, od)

		batch_num = 1
		train_loss_epoch = 0.0
		val_loss_epoch = 0.0

		# Train Mode
		model.train()

		start_time= time()
		total_batches = len(train_dataloader)
		# Batch-wise Training
		for data in train_dataloader:
			# if batch_num % config.display_freq==0:
			# 	od = OrderedDict()
			# 	od['Batch'] = batch_num
			# 	od['Loss'] = loss
			# 	print_log(logger, od)

			#pdb.set_trace()

			ques = data['ques']

			sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
			sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
			sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc1, voc2, device)

			loss = model.trainer(ques, sent2_var, input_len2, config, device, logger)
			train_loss_epoch += loss
			batch_num+=1
			print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

		train_loss_epoch = train_loss_epoch / len(train_dataloader)

		time_taken = (time() - start_time)/60.0

		if writer:
			writer.add_scalar('loss/train_loss', train_loss_epoch, epoch + epoch_offset)

		logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
		logger.debug('Starting Validation')
		# pdb.set_trace()

		val_score_epoch, val_loss_epoch, acc_score = run_validation(config=config, model=model, val_dataloader=val_dataloader, voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch)

		if acc_score > max_acc_score:
			max_acc_score = acc_score

		# bleu_score = val_score_epoch[0]

		if val_score_epoch[0] > max_val_score:
			min_val_loss = val_loss_epoch
			max_val_score = val_score_epoch[0]

			state = {
				'epoch' : epoch + epoch_offset,
				'model_state_dict': model.state_dict(),
				'voc1': model.voc1,
				'voc2': model.voc2,
				'optimizer_state_dict': model.optimizer.state_dict(),
				'train_loss' : train_loss_epoch,
				'val_loss' : min_val_loss,
				'val_bleu_score': max_val_score,
				'val_acc_score': acc_score
			}
			logger.debug('Validation Bleu: {}'.format(val_score_epoch[0]))

			save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)
			estop_count =0
		else:
			estop_count+=1

		if writer:
			writer.add_scalar('loss/val_loss', val_loss_epoch, epoch + epoch_offset)
			writer.add_scalar('acc/val_score', val_score_epoch[0], epoch + epoch_offset)

		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		od['train_loss'] = train_loss_epoch
		od['val_loss']= val_loss_epoch
		od['val_bleu_score'] = max_val_score
		od['val_acc_score'] = acc_score
		od['max_acc'] = max_acc_score
		od['BLEU'] = val_score_epoch
		print_log(logger, od)

		if config.histogram and writer:
			# pdb.set_trace()
			for name, param in model.named_parameters():
				writer.add_histogram(name, param, epoch + epoch_offset)

		if estop_count >config.early_stopping:
			logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			break


	writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
	writer.close()

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.results:
		store_results(config, max_val_score, max_acc_score, min_val_loss, train_loss_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))

	return max_acc_score

def run_validation(config, model, val_dataloader, voc1, voc2, device, logger, epoch_num):
	batch_num =1
	val_loss_epoch =0.0
	val_score_epoch =0.0
	acc_score = 0
	model.eval()

	refs= []
	hyps= []

	display_n = 16 if 16 < config.batch_size else config.batch_size

	with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
		f_out.write('---------------------------------------\n')
		f_out.write('Epoch: ' + str(epoch_num) + '\n')
		f_out.write('---------------------------------------\n')
	total_batches = len(val_dataloader)
	for data in val_dataloader:
		sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
		sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
		nums = data['nums']
		ans = data['ans']

		#pdb.set_trace()

		ques = data['ques']

		sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

		val_loss, decoder_output, decoder_attn = model.greedy_decode(ques, sent2_var, input_len2, validation=True)

		acc_score += cal_score(decoder_output, nums, ans)

		sent1s = idx_to_sents(voc1, sent1_var, no_eos= True)
		sent2s = idx_to_sents(voc2, sent2_var, no_eos= True)

		refs += [[' '.join(sent2s[i])] for i in range(sent2_var.size(1))]
		hyps += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]

		with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
			f_out.write('Batch: ' + str(batch_num) + '\n')
			f_out.write('---------------------------------------\n')
			for i in range(len(sent1s[:display_n])):
				try:
					f_out.write('Example: ' + str(i) + '\n')
					f_out.write('Source: ' + stack_to_string(sent1s[i]) + '\n')
					f_out.write('Target: ' + stack_to_string(sent2s[i]) + '\n')
					f_out.write('Generated: ' + stack_to_string(decoder_output[i]) + '\n' + '\n')
				except:
					logger.warning('Exception: Failed to generate')
					pdb.set_trace()
					break
			f_out.write('---------------------------------------\n')
			f_out.close()

		if batch_num % config.display_freq ==0:
			for i in range(len(sent1s[:display_n])):
				try:
					od = OrderedDict()
					logger.info('-------------------------------------')
					od['Source'] = ' '.join(sent1s[i])

					od['Target'] = ' '.join(sent2s[i])

					od['Generated'] = ' '.join(decoder_output[i])
					print_log(logger, od)
					logger.info('-------------------------------------')
				except:
					logger.warning('Exception: Failed to generate')
					pdb.set_trace()
					break

		val_loss_epoch += val_loss
		batch_num +=1
		print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

	# len(val_dataloader) = len(val_data)/ batch_size
	# val_loss_epoch = val_loss_epoch/len(val_dataloader)
	# pdb.set_trace()
	val_score_epoch = bleu_scorer(refs, hyps)

	return val_score_epoch, val_loss_epoch/len(val_dataloader), acc_score/len(val_dataloader)


def get_hiddens(config, model, val_dataloader, voc1, voc2, device):
	batch_num =1
	
	model.eval()

	hiddens = []
	operands = []

	for data in val_dataloader:
		if len(data['ques']) == config.batch_size:
			sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
			sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
			nums = data['nums']
			ans = data['ans']

			#pdb.set_trace()

			ques = data['ques']

			sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

			hidden, decoder_output = model.obtain_hidden(config, ques, sent2_var, input_len2)

			infix = get_infix_eq(decoder_output, nums)[0] # WORKS ONLY FOR BATCH SIZE 1
			words = infix.split()

			type_rep = []
			operand_types = []

			for w in range(len(words)):
				if words[w] == '/':
					if words[w-1][0] == 'n':
						operand_types.append(['dividend', words[w-1]])
					if words[w+1][0] == 'n':
						operand_types.append(['divisor', words[w+1]])
				elif words[w] == '-':
					if words[w-1][0] == 'n':
						operand_types.append(['minuend', words[w-1]])
					if words[w+1][0] == 'n':
						operand_types.append(['subtrahend', words[w+1]])

			for z in range(len(operand_types)):
				entity = operand_types[z][1]
				for y in range(len(hidden)):
					if hidden[y][0] == entity:
						type_rep.append([operand_types[z][0], hidden[y][1]])

			hiddens = hiddens + hidden
			operands = operands + type_rep

	return hiddens, operands

def decode_greedy(model, sents, config, voc1, voc2, logger, device):

	input_seq, _ , input_len, _ = process_batch(sents, [], voc1, voc2, device)

	decoder_ouput, _ = model.greedy_decode(input_seq)

	outputs= [' '.join(decoder_output[i]) for i in range(len(decoder_output))]

	
	for i in range(len(sents)):
		logger.info('---------------------------------------------------')
		od = OrderedDict()
		od['Source'] = sents[i]
		od['Generated'] = outputs[i]
		print_log(logger, od)
		logger.info('---------------------------------------------------')



