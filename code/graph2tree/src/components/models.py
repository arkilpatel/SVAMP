import torch
import torch.nn as nn
from gensim import models
import torch.nn.functional as F
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pdb

class Embedding(nn.Module):
	def __init__(self, config, input_lang, input_size, embedding_size, dropout=0.5):
		super(Embedding, self).__init__()

		self.config = config
		self.input_lang = input_lang
		self.input_size = input_size
		self.embedding_size = embedding_size

		if self.config.embedding == 'word2vec':
			self.config.embedding_size = 300
			self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		self.em_dropout = nn.Dropout(dropout)

	def _form_embeddings(self, file_path):
		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.input_size, self.config.embedding_size)
		for temp_ind in range(len(self.input_lang.index2word)):
			value = self.input_lang.index2word[temp_ind]
			if value in weights_all:
				weight_req[temp_ind] = torch.FloatTensor(weights_all[value])
		# for key, value in self.voc1.id2w.items():
		# 	if value in weights_all:
		# 		weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req

	def forward(self, input_seqs):
		embedded = self.embedding(input_seqs)  # S x B x E
		embedded = self.em_dropout(embedded)
		return embedded

class EncoderRNN(nn.Module):
	def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
		super(EncoderRNN, self).__init__()

		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		self.em_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

	def forward(self, input_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = self.embedding(input_seqs)  # S x B x E
		embedded = self.em_dropout(embedded)
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
		# S x B x H
		return outputs, hidden


class Attn(nn.Module):
	def __init__(self, hidden_size):
		super(Attn, self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(hidden_size * 2, hidden_size)
		self.score = nn.Linear(hidden_size, 1, bias=False)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, hidden, encoder_outputs, seq_mask=None):
		max_len = encoder_outputs.size(0)
		repeat_dims = [1] * hidden.dim()
		repeat_dims[0] = max_len
		hidden = hidden.repeat(*repeat_dims)  # S x B x H
		# For each position of encoder outputs
		this_batch_size = encoder_outputs.size(1)
		energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
		attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
		attn_energies = attn_energies.squeeze(1)
		attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
		if seq_mask is not None:
			attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
		attn_energies = self.softmax(attn_energies)
		# Normalize energies to weights in range 0 to 1, resize to B x 1 x S
		return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
	def __init__(
			self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
		super(AttnDecoderRNN, self).__init__()

		# Keep for reference
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		self.em_dropout = nn.Dropout(dropout)
		self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		# Choose attention model
		self.attn = Attn(hidden_size)

	def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
		# Get the embedding of the current input word (last output word)
		batch_size = input_seq.size(0)
		embedded = self.embedding(input_seq)
		embedded = self.em_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

		# Calculate attention from current RNN state and all encoder outputs;
		# apply to encoder outputs to get weighted average
		attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

		# Get current hidden state from input word and last hidden state
		rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

		# Attentional vector using the RNN hidden state and context vector
		# concatenated together (Luong eq. 5)
		output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

		# Return final output, hidden state
		return output, hidden


class TreeNode:  # the class save the tree node
	def __init__(self, embedding, left_flag=False):
		self.embedding = embedding
		self.left_flag = left_flag


class Score(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Score, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.attn = nn.Linear(hidden_size + input_size, hidden_size)
		self.score = nn.Linear(hidden_size, 1, bias=False)

	def forward(self, hidden, num_embeddings, num_mask=None):
		max_len = num_embeddings.size(1)
		repeat_dims = [1] * hidden.dim()
		repeat_dims[1] = max_len
		hidden = hidden.repeat(*repeat_dims)  # B x O x H
		# For each position of encoder outputs
		this_batch_size = num_embeddings.size(0)
		energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
		score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
		score = score.squeeze(1)
		score = score.view(this_batch_size, -1)  # B x O
		if num_mask is not None:
			score = score.masked_fill_(num_mask, -1e12)
		return score


class TreeAttn(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(TreeAttn, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.attn = nn.Linear(hidden_size + input_size, hidden_size)
		self.score = nn.Linear(hidden_size, 1)

	def forward(self, hidden, encoder_outputs, seq_mask=None):
		max_len = encoder_outputs.size(0)

		repeat_dims = [1] * hidden.dim()
		repeat_dims[0] = max_len
		hidden = hidden.repeat(*repeat_dims)  # S x B x H
		this_batch_size = encoder_outputs.size(1)

		energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

		score_feature = torch.tanh(self.attn(energy_in))
		attn_energies = self.score(score_feature)  # (S x B) x 1
		attn_energies = attn_energies.squeeze(1)
		attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
		if seq_mask is not None:
			attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
		attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

		return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
	# def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
	def __init__(self, cell_type, embedding_size, hidden_size, n_layers=2, dropout=0.5):
		super(EncoderSeq, self).__init__()

		# self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		# self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		# self.em_dropout = nn.Dropout(dropout)

		if cell_type == 'lstm':
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
							   num_layers=self.n_layers,
							   dropout=(0 if self.n_layers == 1 else self.dropout),
							   bidirectional=True)
		elif cell_type == 'gru':
			self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
		else:
			self.rnn = nn.RNN(self.embedding_size, self.hidden_size,
							  num_layers=self.n_layers,
							  nonlinearity='tanh',							# ['relu', 'tanh']
							  dropout=(0 if self.n_layers == 1 else self.dropout),
							  bidirectional=True)

		self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

	def forward(self, embedded, input_lengths, orig_idx, batch_graph, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		# embedded = self.embedding(input_seqs)  # S x B x E
		# embedded = self.em_dropout(embedded)
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		pade_hidden = hidden
		# pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
		pade_outputs, pade_hidden = self.rnn(packed, pade_hidden)
		pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

		if orig_idx is not None:
			pade_outputs = pade_outputs.index_select(1, orig_idx)

		problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
		pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
		# pdb.set_trace()
		_, pade_outputs = self.gcn(pade_outputs, batch_graph)
		pade_outputs = pade_outputs.transpose(0, 1)
		return pade_outputs, problem_output


class Prediction(nn.Module):
	# a seq2tree decoder with Problem aware dynamic encoding

	def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
		super(Prediction, self).__init__()

		# Keep for reference
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.op_nums = op_nums

		# Define layers
		self.dropout = nn.Dropout(dropout)

		self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

		# for Computational symbols and Generated numbers
		self.concat_l = nn.Linear(hidden_size, hidden_size)
		self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
		self.concat_lg = nn.Linear(hidden_size, hidden_size)
		self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

		self.ops = nn.Linear(hidden_size * 2, op_nums)

		self.attn = TreeAttn(hidden_size, hidden_size)
		self.score = Score(hidden_size * 2, hidden_size)

	def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
		current_embeddings = []

		for st in node_stacks:
			if len(st) == 0:
				current_embeddings.append(padding_hidden)
			else:
				current_node = st[-1]
				current_embeddings.append(current_node.embedding)

		current_node_temp = []
		for l, c in zip(left_childs, current_embeddings):
			if l is None:
				c = self.dropout(c)
				g = torch.tanh(self.concat_l(c))
				t = torch.sigmoid(self.concat_lg(c))
				current_node_temp.append(g * t)
			else:
				ld = self.dropout(l)
				c = self.dropout(c)
				g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
				t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
				current_node_temp.append(g * t)

		current_node = torch.stack(current_node_temp)

		current_embeddings = self.dropout(current_node)

		current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
		current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

		# the information to get the current quantity
		batch_size = current_embeddings.size(0)
		# predict the output (this node corresponding to output(number or operator)) with PADE

		repeat_dims = [1] * self.embedding_weight.dim()
		repeat_dims[0] = batch_size
		embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
		embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

		leaf_input = torch.cat((current_node, current_context), 2)
		leaf_input = leaf_input.squeeze(1)
		leaf_input = self.dropout(leaf_input)

		# p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
		# max pooling the embedding_weight
		embedding_weight_ = self.dropout(embedding_weight)
		num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

		# num_score = nn.functional.softmax(num_score, 1)

		op = self.ops(leaf_input)

		# return p_leaf, num_score, op, current_embeddings, current_attn

		return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
	def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
		super(GenerateNode, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size

		self.embeddings = nn.Embedding(op_nums, embedding_size)
		self.em_dropout = nn.Dropout(dropout)
		self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
		self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
		self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
		self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

	def forward(self, node_embedding, node_label, current_context):
		node_label_ = self.embeddings(node_label)
		node_label = self.em_dropout(node_label_)
		node_embedding = node_embedding.squeeze(1)
		current_context = current_context.squeeze(1)
		node_embedding = self.em_dropout(node_embedding)
		current_context = self.em_dropout(current_context)

		l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
		l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
		r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
		r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
		l_child = l_child * l_child_g
		r_child = r_child * r_child_g
		return l_child, r_child, node_label_


class Merge(nn.Module):
	def __init__(self, hidden_size, embedding_size, dropout=0.5):
		super(Merge, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size

		self.em_dropout = nn.Dropout(dropout)
		self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
		self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

	def forward(self, node_embedding, sub_tree_1, sub_tree_2):
		sub_tree_1 = self.em_dropout(sub_tree_1)
		sub_tree_2 = self.em_dropout(sub_tree_2)
		node_embedding = self.em_dropout(node_embedding)

		sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
		sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
		sub_tree = sub_tree * sub_tree_g
		return sub_tree
	
	
	
# Graph Module
def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff,d_out, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_out)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Graph_Module(nn.Module):
	def __init__(self, indim, hiddim, outdim, dropout=0.3):
		super(Graph_Module, self).__init__()
		'''
		## Variables:
		- indim: dimensionality of input node features
		- hiddim: dimensionality of the joint hidden embedding
		- outdim: dimensionality of the output node features
		- combined_feature_dim: dimensionality of the joint hidden embedding for graph
		- K: number of graph nodes/objects on the image
		'''
		self.in_dim = indim
		#self.combined_dim = outdim
		
		#self.edge_layer_1 = nn.Linear(indim, outdim)
		#self.edge_layer_2 = nn.Linear(outdim, outdim)
		
		#self.dropout = nn.Dropout(p=dropout)
		#self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
		#self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)
		self.h = 4
		self.d_k = outdim//self.h
		
		#layer = GCN(indim, hiddim, self.d_k, dropout)
		self.graph = clones(GCN(indim, hiddim, self.d_k, dropout), 4)
		
		#self.Graph_0 = GCN(indim, hiddim, outdim//4, dropout)
		#self.Graph_1 = GCN(indim, hiddim, outdim//4, dropout)
		#self.Graph_2 = GCN(indim, hiddim, outdim//4, dropout)
		#self.Graph_3 = GCN(indim, hiddim, outdim//4, dropout)
		
		self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
		self.norm = LayerNorm(outdim)

	def get_adj(self, graph_nodes):
		'''
		## Inputs:
		- graph_nodes (batch_size, K, in_feat_dim): input features
		## Returns:
		- adjacency matrix (batch_size, K, K)
		'''
		self.K = graph_nodes.size(1)
		graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)
		
		# layer 1
		h = self.edge_layer_1(graph_nodes)
		h = F.relu(h)
		
		# layer 2
		h = self.edge_layer_2(h)
		h = F.relu(h)

		# outer product
		h = h.view(-1, self.K, self.combined_dim)
		adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
		
		adjacency_matrix = self.b_normal(adjacency_matrix)

		return adjacency_matrix
	
	def normalize(self, A, symmetric=True):
		'''
		## Inputs:
		- adjacency matrix (K, K) : A
		## Returns:
		- adjacency matrix (K, K) 
		'''
		A = A + torch.eye(A.size(0)).cuda().float()
		d = A.sum(1)
		if symmetric:
			# D = D^{-1/2}
			D = torch.diag(torch.pow(d, -0.5))
			return D.mm(A).mm(D)
		else :
			D = torch.diag(torch.pow(d,-1))
			return D.mm(A)
	   
	def b_normal(self, adj):
		batch = adj.size(0)
		for i in range(batch):
			adj[i] = self.normalize(adj[i])
		return adj

	def forward(self, graph_nodes, graph):
		'''
		## Inputs:
		- graph_nodes (batch_size, K, in_feat_dim): input features
		## Returns:
		- graph_encode_features (batch_size, K, out_feat_dim)
		'''
		nbatches = graph_nodes.size(0)
		mbatches = graph.size(0)
		if nbatches != mbatches:
			graph_nodes = graph_nodes.transpose(0, 1)
		# adj (batch_size, K, K): adjacency matrix
		if not bool(graph.numel()):
			adj = self.get_adj(graph_nodes)
			#adj = adj.unsqueeze(1)
			#adj = torch.cat((adj,adj,adj),1)
			adj_list = [adj,adj,adj,adj]
		else:
			adj = graph.float()
			adj_list = [adj[:,1,:],adj[:,1,:],adj[:,4,:],adj[:,4,:]]
		#print(adj)
		
		g_feature = \
			tuple([l(graph_nodes,x) for l, x in zip(self.graph,adj_list)])
		#g_feature_0 = self.Graph_0(graph_nodes,adj[0])
		#g_feature_1 = self.Graph_1(graph_nodes,adj[1])
		#g_feature_2 = self.Graph_2(graph_nodes,adj[2])
		#g_feature_3 = self.Graph_3(graph_nodes,adj[3])
		#print('g_feature')
		#print(type(g_feature))
		
		
		g_feature = self.norm(torch.cat(g_feature,2)) + graph_nodes
		#print('g_feature')
		#print(g_feature.shape)
		
		graph_encode_features = self.feed_foward(g_feature) + g_feature
		
		return adj, graph_encode_features

# GCN
class GCN(nn.Module):
	def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
		super(GCN, self).__init__()
		'''
		## Inputs:
		- graph_nodes (batch_size, K, in_feat_dim): input features
		- adjacency matrix (batch_size, K, K)
		## Returns:
		- gcn_enhance_feature (batch_size, K, out_feat_dim)
		'''
		self.gc1 = GraphConvolution(in_feat_dim, nhid)
		self.gc2 = GraphConvolution(nhid, out_feat_dim)
		self.dropout = dropout

	def forward(self, x, adj):
		x = F.relu(self.gc1(x, adj))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, adj)
		return x
	
# Graph_Conv
class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		#print(input.shape)
		#print(self.weight.shape)
		support = torch.matmul(input, self.weight)
		#print(adj.shape)
		#print(support.shape)
		output = torch.matmul(adj, support)
		
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'
