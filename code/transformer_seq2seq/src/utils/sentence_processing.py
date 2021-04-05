import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def sent_to_idx(voc, sent, max_length, flag = 0):
	if flag == 0:
		idx_vec = []
	else:
		idx_vec = [voc.get_id('<s>')]
	for w in sent.split(' '):
		try:
			idx = voc.get_id(w)
			idx_vec.append(idx)
		except:
			idx_vec.append(voc.get_id('unk'))
	# idx_vec.append(voc.get_id('</s>'))
	if flag == 1 and len(idx_vec) < max_length-1:
		idx_vec.append(voc.get_id('</s>'))
	return idx_vec

def sents_to_idx(voc, sents, max_length, flag = 0):
	all_indexes = []
	for sent in sents:
		all_indexes.append(sent_to_idx(voc, sent, max_length, flag))
	return all_indexes

def sent_to_tensor(voc, sentence, device, max_length):
	indexes = sent_to_idx(voc, sentence, max_length)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def batch_to_tensor(voc, sents, device, max_length):
	batch_sent = []
	# batch_label = []
	for sent in sents:
		sent_id = sent_to_tensor(voc, sent, device, max_length)
		batch_sent.append(sent_id)

	return batch_sent

def idx_to_sent(voc, tensor, no_eos=False):
	sent_word_list = []
	for idx in tensor:
		word = voc.get_word(idx.item())
		if no_eos:
			if word != '</s>':
				sent_word_list.append(word)
			# else:
			# 	break
		else:
			sent_word_list.append(word)
	return sent_word_list

def idx_to_sents(voc, tensors, no_eos=False):
	tensors = tensors.transpose(0, 1)
	batch_word_list = []
	for tensor in tensors:
		batch_word_list.append(idx_to_sent(voc, tensor, no_eos))

	return batch_word_list

def pad_seq(seq, max_length, voc):
	seq += [voc.get_id('</s>') for i in range(max_length - len(seq))]
	return seq

def sort_by_len(seqs, input_len, device=None, dim=1):
	orig_idx = list(range(seqs.size(dim)))

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
	return sorted_seqs, sorted_lens, orig_idx

def restore_order(seqs, input_len, orig_idx):
	orig_seqs= [seqs[i] for i in orig_idx]
	orig_lens= [input_len[i] for i in orig_idx]
	return orig_seqs, orig_lens

def process_batch(sent1s, sent2s, voc1, voc2, device):
	input_len1 = [len(s) for s in sent1s]
	input_len2 = [len(s) for s in sent2s]
	max_length_1 = max(input_len1)
	max_length_2 = max(input_len2)

	sent1s_padded = [pad_seq(s, max_length_1, voc1) for s in sent1s]
	sent2s_padded = [pad_seq(s, max_length_2, voc2) for s in sent2s]

	# Convert to [Max_len X Batch]
	sent1_var = Variable(torch.LongTensor(sent1s_padded)).transpose(0, 1)
	sent2_var = Variable(torch.LongTensor(sent2s_padded)).transpose(0, 1)

	sent1_var = sent1_var.to(device)
	sent2_var = sent2_var.to(device)

	return sent1_var, sent2_var, input_len1, input_len2
