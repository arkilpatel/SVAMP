import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DecoderRNN(nn.Module):
	'''
	To DO
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
	def __init__(self, embedding, cell_type, hidden_size, output_size, nlayers=1, dropout=0.2):
		super(DecoderRNN, self).__init__()
		self.hidden_size        = hidden_size
		self.cell_type          = cell_type
		self.embedding          = embedding
		self.embedding_size     = self.embedding.embedding_dim
		self.embedding_dropout = nn.Dropout(dropout)
		self.nlayers            = nlayers
		self.output_size        = output_size

		if self.cell_type == 'lstm':
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.nlayers, dropout=(0 if nlayers == 1 else dropout))
		else:
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size, num_layers=self.nlayers, dropout=(0 if nlayers == 1 else dropout))

		self.out     = nn.Linear(self.hidden_size, self.output_size)


	def forward(self, input_step, last_hidden):
		'''
		To Do
			Args:
				input_seqs (tensor) : input tensor | size : [Seq_len X Batch_size]
				input_lengths (list/tensor) : length of each input sentence | size : [Batch_size] 
				device (gpu) : Used for sorting the sentences and putting it to device

			Returns:
				output (tensor) : Last State representations of RNN [Seq_len X Batch_size X hidden_size]
				hidden (tuple)	: Hidden states and (cell states) of recurrent networks
		'''
		output              = self.embedding(input_step)
		output              = self.embedding_dropout(output)
		output              = output.view(1, input_step.size(0), self.embedding_size)
		output              = F.relu(output)
		output, last_hidden = self.rnn(output, last_hidden)
		output              = output.squeeze(0)
		output              = self.out(output)
		output              = F.log_softmax(output, dim=1)

		return output, last_hidden
