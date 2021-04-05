import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# Luong attention layer
class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method.")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def dot_score(self, hidden, encoder_outputs):
		return torch.sum(hidden * encoder_outputs, dim=2)

	def general_score(self, hidden, encoder_outputs):
		energy = self.attn(encoder_outputs)
		return torch.sum(hidden * energy, dim=2)

	def concat_score(self, hidden, encoder_outputs):
		energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)

	def forward(self, hidden, encoder_outputs):
		# Calculate the attention weights (energies) based on the given method
		if self.method == 'general':
			attn_energies = self.general_score(hidden, encoder_outputs)
		elif self.method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_outputs)
		elif self.method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_outputs)

		# Transpose max_length and batch_size dimensions
		attn_energies = attn_energies.t()

		# Return the softmax normalized probability scores (with added dimension)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, embedding, cell_type, hidden_size, output_size, nlayers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model 	= attn_model
		self.hidden_size 	= hidden_size
		self.output_size 	= output_size
		self.nlayers 		= nlayers
		self.dropout 		= dropout
		self.cell_type 		= cell_type

		# Define layers
		self.embedding = embedding
		self.embedding_size  = self.embedding.embedding_dim
		self.embedding_dropout = nn.Dropout(self.dropout)
		if self.cell_type == 'gru':
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
		else:
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
		self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

		self.attn = Attn(self.attn_model, self.hidden_size)

	def forward(self, input_step, last_hidden, encoder_outputs):
		# Note: we run this one step (word) at a time
		# Get embedding of current input word
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)

		try:
			embedded = embedded.view(1, input_step.size(0), self.embedding_size)
		except:
			embedded = embedded.view(1, 1, self.embedding_size)

		rnn_output, hidden = self.rnn(embedded, last_hidden)
		# Calculate attention weights from the current GRU output
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# Multiply attention weights to encoder outputs to get new "weighted sum" context vector
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		# Concatenate weighted context vector and GRU output using Luong eq. 5
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = F.relu(self.concat(concat_input))
		representation = concat_output
		# Predict next word using Luong eq. 6
		output = self.out(concat_output)
		output = F.log_softmax(output, dim=1)
		# Return output and final hidden state
		return output, hidden, attn_weights, representation