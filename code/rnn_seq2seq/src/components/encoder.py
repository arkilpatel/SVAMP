import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from src.utils.sentence_processing import sort_by_len, restore_order

class Encoder(nn.Module):
	'''
	Encoder helps in building the sentence encoding module for a batched version
	of data that is sent in [T x B] having corresponding input lengths in [1 x B]

	Args:
			hidden_size: Hidden size of the RNN cell
			embedding: Embeddings matrix [vocab_size, embedding_dim]
			cell_type: Type of RNN cell to be used : LSTM, GRU
			nlayers: Number of layers of LSTM (default = 1)
			dropout: Dropout Rate (default = 0.1)
			bidirectional: Bidirectional model to be formed (default: False)
	'''

	def __init__(self, hidden_size=512,embedding_size = 768, cell_type='lstm', nlayers=1, dropout=0.1, bidirectional=True):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.nlayers = nlayers
		self.dropout = dropout
		self.cell_type = cell_type
		self.embedding_size = embedding_size
		# self.embedding_size = self.embedding.embedding_dim
		self.bidirectional = bidirectional

		if self.cell_type == 'lstm':
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
							   num_layers=self.nlayers,
							   dropout=(0 if self.nlayers == 1 else dropout),
							   bidirectional=bidirectional)
		elif self.cell_type == 'gru':
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size,
							  num_layers=self.nlayers,
							  dropout=(0 if self.nlayers == 1 else dropout),
							  bidirectional=bidirectional)
		else:
			self.rnn = nn.RNN(self.embedding_size, self.hidden_size,
							  num_layers=self.nlayers,
							  nonlinearity='tanh',							# ['relu', 'tanh']
							  dropout=(0 if self.nlayers == 1 else dropout),
							  bidirectional=bidirectional)

	def forward(self, sorted_seqs, sorted_len, orig_idx, device=None, hidden=None):
		'''
			Args:
				input_seqs (tensor) : input tensor | size : [Seq_len X Batch_size]
				input_lengths (list/tensor) : length of each input sentence | size : [Batch_size] 
				device (gpu) : Used for sorting the sentences and putting it to device

			Returns:
				output (tensor) : Last State representations of RNN [Seq_len X Batch_size X hidden_size]
				hidden (tuple)	: Hidden states and (cell states) of recurrent networks
		'''

		# sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seqs, input_lengths, device)
		# pdb.set_trace()

		#embedded = self.embedding(sorted_seqs)  ### NO MORE IDS
		packed = torch.nn.utils.rnn.pack_padded_sequence(
			sorted_seqs, sorted_len)
		outputs, hidden = self.rnn(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
			outputs)  # unpack (back to padded)

		outputs = outputs.index_select(1, orig_idx)

		if self.bidirectional:
			outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

		return outputs, hidden
