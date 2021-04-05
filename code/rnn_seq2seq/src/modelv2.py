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
from transformers import AdamW
# from tensorboardX import SummaryWriter
from gensim import models
from src.components.encoder import Encoder
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
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

		self.embedding2 = nn.Embedding(self.voc2.nwords, self.config.emb2_size)
		nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)

		if self.config.embedding == 'bert':
			self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			self.config.emb1_size = 300
			self.embedding1 = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.emb1_size)
			nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

		self.logger.debug('Building Encoders...')
		self.encoder = Encoder(
			self.config.hidden_size,
			self.config.emb1_size,
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
		self.params =   list(self.embedding1.parameters()) + \
						list(self.encoder.parameters()) + \
						list(self.decoder.parameters())

		if self.config.separate_opt:
			self.emb_optimizer = AdamW(self.embedding1.parameters(), lr = self.config.emb_lr, correct_bias = True)
			self.optimizer = optim.Adam(
				[{"params": self.encoder.parameters()},
				{"params": self.decoder.parameters()}],
				lr = self.config.lr,
			)
		else:
			if self.config.opt == 'adam':
				self.optimizer = optim.Adam(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)
			elif self.config.opt == 'adadelta':
				self.optimizer = optim.Adadelta(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)
			elif self.config.opt == 'asgd':
				self.optimizer = optim.ASGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)
			else:
				self.optimizer = optim.SGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)

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

	def trainer(self, ques, input_seq1, input_seq2, input_len1, input_len2, config, device=None ,logger=None):
		'''
			Args:
				ques (list): input examples as is (i.e. not indexed) | size : [batch_size]
			Returns:
				
		'''
		self.optimizer.zero_grad()
		if self.config.separate_opt:
			self.emb_optimizer.zero_grad()

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			input_seq1, input_len1 = self.embedding1(ques)
			input_seq1 = input_seq1.transpose(0,1)
			# input_seq1: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			# sorted_seqs: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			# orig_idx: Tensor [BS]
		else:
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			sorted_seqs = self.embedding1(sorted_seqs)

		encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)
		
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
		if self.config.separate_opt:
			self.emb_optimizer.step()

		return self.loss.item()/target_len

	def greedy_decode(self, ques, input_seq1=None, input_seq2=None, input_len1=None, input_len2=None, validation=False, return_probs = False):
		with torch.no_grad():
			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				input_seq1, input_len1 = self.embedding1(ques)
				input_seq1 = input_seq1.transpose(0,1)
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			else:
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
				sorted_seqs = self.embedding1(sorted_seqs)

			encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)

			loss = 0.0
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

	def obtain_hidden(self, config, ques, input_seq1=None, input_seq2=None, input_len1=None, input_len2=None):
		with torch.no_grad():
			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				input_seq1, input_len1 = self.embedding1(ques)
				input_seq1 = input_seq1.transpose(0,1)
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			else:
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
				sorted_seqs = self.embedding1(sorted_seqs)

			encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)

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

def train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, epoch_offset= 0, min_val_loss=float('inf'), max_val_bleu=0.0, max_val_acc = 0.0, min_train_loss=float('inf'), max_train_acc = 0.0, best_epoch = 0, writer= None):
	'''
		Add Docstring
	'''

	if config.histogram and config.save_writer and writer:
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch_offset)
	
	estop_count=0
	
	for epoch in range(1, config.epochs + 1):
		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		print_log(logger, od)

		batch_num = 1
		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		train_acc_epoch_cnt = 0.0
		train_acc_epoch_tot = 0.0
		val_loss_epoch = 0.0

		start_time= time()
		total_batches = len(train_dataloader)

		for data in train_dataloader:
			ques = data['ques']

			sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
			sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
			sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc1, voc2, device)

			nums = data['nums']
			ans = data['ans']

			model.train()

			loss = model.trainer(ques, sent1_var, sent2_var, input_len1, input_len2, config, device, logger)
			train_loss_epoch += loss

			if config.show_train_acc:
				model.eval()

				_, decoder_output, _ = model.greedy_decode(ques, sent1_var, sent2_var, input_len1, input_len2, validation=True)
				temp_acc_cnt, temp_acc_tot, _ = cal_score(decoder_output, nums, ans, data['eqn'])
				train_acc_epoch_cnt += temp_acc_cnt
				train_acc_epoch_tot += temp_acc_tot

			batch_num+=1
			print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

		train_loss_epoch = train_loss_epoch/len(train_dataloader)
		if config.show_train_acc:
			train_acc_epoch = train_acc_epoch_cnt/train_acc_epoch_tot
		else:
			train_acc_epoch = 0.0

		time_taken = (time() - start_time)/60.0

		if config.save_writer and writer:
			writer.add_scalar('loss/train_loss', train_loss_epoch, epoch + epoch_offset)

		logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
		logger.debug('Starting Validation')

		val_bleu_epoch, val_loss_epoch, val_acc_epoch = run_validation(config=config, model=model, dataloader=val_dataloader, voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch)

		if train_loss_epoch < min_train_loss:
			min_train_loss = train_loss_epoch

		if train_acc_epoch > max_train_acc:
			max_train_acc = train_acc_epoch

		if val_bleu_epoch[0] > max_val_bleu:
			max_val_bleu = val_bleu_epoch[0]

		if val_loss_epoch < min_val_loss:
			min_val_loss = val_loss_epoch

		if val_acc_epoch > max_val_acc:
			max_val_acc = val_acc_epoch
			best_epoch = epoch + epoch_offset

			if config.separate_opt:
				state = {
					'epoch' : epoch + epoch_offset,
					'best_epoch': best_epoch,
					'model_state_dict': model.state_dict(),
					'voc1': model.voc1,
					'voc2': model.voc2,
					'optimizer_state_dict': model.optimizer.state_dict(),
					'emb_optimizer_state_dict': model.emb_optimizer.state_dict(),
					'train_loss_epoch' : train_loss_epoch,
					'min_train_loss' : min_train_loss,
					'train_acc_epoch' : train_acc_epoch,
					'max_train_acc' : max_train_acc,
					'val_loss_epoch' : val_loss_epoch,
					'min_val_loss' : min_val_loss,
					'val_acc_epoch' : val_acc_epoch,
					'max_val_acc' : max_val_acc,
					'val_bleu_epoch': val_bleu_epoch[0],
					'max_val_bleu': max_val_bleu
				}
			else:
				state = {
					'epoch' : epoch + epoch_offset,
					'best_epoch': best_epoch,
					'model_state_dict': model.state_dict(),
					'voc1': model.voc1,
					'voc2': model.voc2,
					'optimizer_state_dict': model.optimizer.state_dict(),
					'train_loss_epoch' : train_loss_epoch,
					'min_train_loss' : min_train_loss,
					'train_acc_epoch' : train_acc_epoch,
					'max_train_acc' : max_train_acc,
					'val_loss_epoch' : val_loss_epoch,
					'min_val_loss' : min_val_loss,
					'val_acc_epoch' : val_acc_epoch,
					'max_val_acc' : max_val_acc,
					'val_bleu_epoch': val_bleu_epoch[0],
					'max_val_bleu': max_val_bleu
				}
			logger.debug('Validation Bleu: {}'.format(val_bleu_epoch[0]))

			if config.save_model:
				save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)
			estop_count = 0
		else:
			estop_count+=1

		if config.save_writer and writer:
			writer.add_scalar('loss/val_loss', val_loss_epoch, epoch + epoch_offset)
			writer.add_scalar('acc/val_score', val_bleu_epoch[0], epoch + epoch_offset)

		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		od['best_epoch'] = best_epoch
		od['train_loss_epoch'] = train_loss_epoch
		od['min_train_loss'] = min_train_loss
		od['val_loss_epoch']= val_loss_epoch
		od['min_val_loss']= min_val_loss
		od['train_acc_epoch'] = train_acc_epoch
		od['max_train_acc'] = max_train_acc
		od['val_acc_epoch'] = val_acc_epoch
		od['max_val_acc'] = max_val_acc
		od['val_bleu_epoch'] = val_bleu_epoch
		od['max_val_bleu'] = max_val_bleu
		print_log(logger, od)

		if config.histogram and config.save_writer and writer:
			for name, param in model.named_parameters():
				writer.add_histogram(name, param, epoch + epoch_offset)

		if estop_count > config.early_stopping:
			logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			break

	if config.save_writer:
		writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
		writer.close()

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.results:
		store_results(config, max_val_bleu, max_val_acc, min_val_loss, max_train_acc, min_train_loss, best_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))

	return max_val_acc

def run_validation(config, model, dataloader, voc1, voc2, device, logger, epoch_num):
	batch_num = 1
	val_loss_epoch = 0.0
	val_bleu_epoch = 0.0
	val_acc_epoch = 0.0
	val_acc_epoch_cnt = 0.0
	val_acc_epoch_tot = 0.0

	model.eval()

	refs= []
	hyps= []

	if config.mode == 'test':
		questions, gen_eqns, act_eqns, scores = [], [], [], []

	display_n = config.batch_size

	with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
		f_out.write('---------------------------------------\n')
		f_out.write('Epoch: ' + str(epoch_num) + '\n')
		f_out.write('---------------------------------------\n')
	total_batches = len(dataloader)
	for data in dataloader:
		sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
		sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
		nums = data['nums']
		ans = data['ans']
		if config.grade_disp:
			grade = data['grade']
		if config.type_disp:
			type1 = data['type']
		if config.challenge_disp:
			type1 = data['type']
			var_type = data['var_type']
			annotator = data['annotator']
			alternate = data['alternate']

		ques = data['ques']

		sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

		val_loss, decoder_output, decoder_attn = model.greedy_decode(ques, sent1_var, sent2_var, input_len1, input_len2, validation=True)

		temp_acc_cnt, temp_acc_tot, disp_corr = cal_score(decoder_output, nums, ans, data['eqn'])
		val_acc_epoch_cnt += temp_acc_cnt
		val_acc_epoch_tot += temp_acc_tot

		sent1s = idx_to_sents(voc1, sent1_var, no_eos= True)
		sent2s = idx_to_sents(voc2, sent2_var, no_eos= True)

		refs += [[' '.join(sent2s[i])] for i in range(sent2_var.size(1))]
		hyps += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]

		if config.mode == 'test':
			questions+= data['ques']
			gen_eqns += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]
			act_eqns += [' '.join(sent2s[i]) for i in range(sent2_var.size(1))]
			scores   += [cal_score([decoder_output[i]], [nums[i]], [ans[i]], [data['eqn'][i]])[0] for i in range(sent1_var.size(1))]

		with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
			f_out.write('Batch: ' + str(batch_num) + '\n')
			f_out.write('---------------------------------------\n')
			for i in range(len(sent1s[:display_n])):
				try:
					f_out.write('Example: ' + str(i) + '\n')
					if config.grade_disp:
						f_out.write('Grade: ' + str(grade[i].item()) + '\n')
					if config.type_disp:
						f_out.write('Type: ' + str(type1[i]) + '\n')
					f_out.write('Source: ' + stack_to_string(sent1s[i]) + '\n')
					f_out.write('Target: ' + stack_to_string(sent2s[i]) + '\n')
					f_out.write('Generated: ' + stack_to_string(decoder_output[i]) + '\n')
					if config.challenge_disp:
						f_out.write('Type: ' + str(type1[i]) + '\n')
						f_out.write('Variation Type: ' + str(var_type[i]) + '\n')
						f_out.write('Annotator: ' + str(annotator[i]) + '\n')
						f_out.write('Alternate: ' + str(alternate[i].item()) + '\n')
					if config.nums_disp:
						src_nums = 0
						tgt_nums = 0
						pred_nums = 0
						for k in range(len(sent1s[i])):
							if sent1s[i][k][:6] == 'number':
								src_nums += 1
						for k in range(len(sent2s[i])):
							if sent2s[i][k][:6] == 'number':
								tgt_nums += 1
						for k in range(len(decoder_output[i])):
							if decoder_output[i][k][:6] == 'number':
								pred_nums += 1
						f_out.write('Numbers in question: ' + str(src_nums) + '\n')
						f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
						f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
					f_out.write('Result: ' + str(disp_corr[i]) + '\n' + '\n')
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

	val_bleu_epoch = bleu_scorer(refs, hyps)
	if config.mode == 'test':
		results_df = pd.DataFrame([questions, act_eqns, gen_eqns, scores]).transpose()
		results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score']
		csv_file_path = os.path.join(config.outputs_path, config.dataset+'.csv')
		results_df.to_csv(csv_file_path, index = False)
		return sum(scores)/len(scores)

	val_acc_epoch = val_acc_epoch_cnt/val_acc_epoch_tot

	return val_bleu_epoch, val_loss_epoch/len(dataloader), val_acc_epoch

def estimate_confidence(config, model, dataloader, logger):
	
	questions	= []
	act_eqns 	= []
	gen_eqns	= []
	scores		= []
	confs		= []
	batch_num = 0
	
	#Load training data (Will be useful for similarity based methods)
	train_df 	= pd.read_csv(os.path.join('data',config.dataset,'train.csv'))
	train_ques	= train_df['Question'].values 
	
	total_batches = len(dataloader)
	logger.info("Beginning estimating confidence based on {} criteria".format(config.conf))
	start = time()
	for data in dataloader:
		ques, eqn, nums, ans = data['ques'], data['eqn'], data['nums'], data['ans']
		
		if config.conf == 'posterior':
			decoded_words, confidence = posterior_based_conf(ques, model)
		elif config.conf == 'similarity':
			decoded_words, confidence = similarity_based_conf(ques, train_ques, model, sim_criteria= config.sim_criteria)
		else:
			#TODO: Implement other methods
			raise ValueError("Other confidence methods not implemented yet. Use -conf posterior")
		
		if not config.adv:
			correct_or_not = [cal_score([decoded_words[i]], [nums[i]], [ans[i]])[0] for i in range(len(decoded_words))]
		else:
			correct_or_not = [-1 for i in range(len(decoded_words))]

		gen_eqn = [' '.join(words) for words in decoded_words]
		
		questions 	+= ques
		act_eqns	+= eqn
		gen_eqns	+= gen_eqn
		scores		+= correct_or_not
		confs		+= list(confidence)
		batch_num	+= 1
		print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

	results_df = pd.DataFrame([questions, act_eqns, gen_eqns, scores, confs]).transpose()
	results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score', 'Confidence']
	if config.conf != 'similarity':
		csv_file_path = os.path.join('ConfidenceEstimates',config.dataset + '_' + config.run_name + '_' + config.conf + '.csv')
	else:
		csv_file_path = os.path.join('ConfidenceEstimates',config.dataset + '_' + config.run_name + '_' + config.conf + '_' + config.sim_criteria + '.csv')
	results_df.to_csv(csv_file_path)
	logger.info("Done in {} seconds".format(time() - start))

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

			ques = data['ques']

			sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

			hidden, decoder_output = model.obtain_hidden(config, ques, sent1_var, sent2_var, input_len1, input_len2)

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


