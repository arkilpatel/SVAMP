import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import pdb

class BertEncoder(nn.Module):
	def __init__(self, bert_model = 'bert-base-uncased',device = 'cuda:0 ', freeze_bert = False):
		super(BertEncoder, self).__init__()
		self.bert_layer = BertModel.from_pretrained(bert_model)
		self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
		self.device = device
		
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False
		
	def bertify_input(self, sentences):
		'''
		Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into BERT
		all_tokens  = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]
		
		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths = self.bertify_input(sentences)

		#Feed through bert
		cont_reps, _ = self.bert_layer(token_ids, attention_mask = attn_masks)

		return cont_reps, input_lengths

class RobertaEncoder(nn.Module):
	def __init__(self, roberta_model = 'roberta-base', device = 'cuda:0 ', freeze_roberta = False):
		super(RobertaEncoder, self).__init__()
		self.roberta_layer = RobertaModel.from_pretrained(roberta_model)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
		self.device = device
		
		if freeze_roberta:
			for p in self.roberta_layer.parameters():
				p.requires_grad = False
		
	def robertify_input(self, sentences):
		'''
		Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids

		'''
		# Tokenize the input sentences for feeding into RoBERTa
		all_tokens  = [['<s>'] + self.roberta_tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]
		
		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.roberta_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		# Obtain attention masks
		pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
		'''
		# Preprocess sentences
		token_ids, attn_masks, input_lengths = self.robertify_input(sentences)

		# Feed through RoBERTa
		cont_reps, _ = self.roberta_layer(token_ids, attention_mask = attn_masks)

		return cont_reps, input_lengths