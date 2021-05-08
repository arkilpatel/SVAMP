import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')

	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test', 'conf'], help='Modes: train, test, conf')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)

	# Run Config
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='mawps', help='Dataset')
	parser.add_argument('-display_freq', type=int, default= 10000, help='number of batches after which to display samples')
	parser.add_argument('-outputs', dest='outputs', action='store_true', help='Show full validation outputs')
	parser.add_argument('-no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
	parser.set_defaults(outputs=True)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Meta Attributes
	parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')
	parser.add_argument('-histogram', dest='histogram', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-histogram', dest='histogram', action='store_false', help='Operate in normal mode')
	parser.set_defaults(histogram=True)
	parser.add_argument('-save_writer', dest='save_writer',action='store_true', help='To write tensorboard')
	parser.add_argument('-no-save_writer', dest='save_writer', action='store_false', help='Dont write tensorboard')
	parser.set_defaults(save_writer=False)

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=2, help='Specify the gpu to use')
	parser.add_argument('-early_stopping', type=int, default=50, help='Early Stopping after n epoch')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=False)
	# parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

	# LSTM parameters
	parser.add_argument('-emb2_size', type=int, default=16, help='Embedding dimensions of inputs')
	parser.add_argument('-cell_type', type=str, default='lstm', help='RNN cell for encoder and decoder, default: lstm')

	parser.add_argument('-use_attn', dest='use_attn',action='store_true', help='To use attention mechanism?')
	parser.add_argument('-no-attn', dest='use_attn', action='store_false', help='Not to use attention mechanism?')
	parser.set_defaults(use_attn=True)

	parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
	parser.add_argument('-hidden_size', type=int, default=256, help='Number of hidden units in each layer')
	parser.add_argument('-depth', type=int, default=1, help='Number of layers in each encoder and decoder')
	parser.add_argument('-dropout', type=float, default=0.1, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	parser.add_argument('-max_length', type=int, default=100, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	parser.add_argument('-bidirectional', dest='bidirectional', action='store_true', help='Bidirectionality in LSTMs')
	parser.add_argument('-no-bidirectional', dest='bidirectional', action='store_false', help='Bidirectionality in LSTMs')
	parser.set_defaults(bidirectional=True)
	parser.add_argument('-lr', type=float, default=0.0005, help='Learning rate')
	# parser.add_argument('-bert_lr', type=float, default=5e-5, help='Larning rate to train BERT embeddings')
	parser.add_argument('-warmup', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for')
	parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=8, help='Batch size')
	parser.add_argument('-epochs', type=int, default=50, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
	parser.add_argument('-separate_opt', dest='separate_opt', action='store_true', help='Separate Optimizers for Embedding and model - AdamW for emb and Adam for model')
	parser.add_argument('-no-separate_opt', dest='separate_opt', action='store_false', help='Common optimizer for Embedding and model')
	parser.set_defaults(separate_opt=False)
	parser.add_argument('-teacher_forcing_ratio', type=float, default=0.9, help='Teacher forcing ratio')

	# Embeddings
	parser.add_argument('-embedding', type=str, default='roberta', choices=['bert', 'roberta', 'word2vec', 'random'], help='Embeddings')
	# parser.add_argument('-use_word2vec', dest='use_word2vec', action='store_true', help='use word2vec')
	# parser.add_argument('-no-use_word2vec', dest='use_word2vec', action='store_false', help='Do not use word2vec')
	# parser.set_defaults(use_word2vec=False)
	# parser.add_argument('-word2vec_bin', type=str, default='/datadrive/satwik/global_data/glove.840B.300d.txt', help='Binary file of word2vec')
	parser.add_argument('-word2vec_bin', type=str, default='/datadrive/global_files/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')
	# parser.add_argument('-train_word2vec', dest='train_word2vec', action='store_true', help='train word2vec')
	# parser.add_argument('-no-train_word2vec', dest='train_word2vec', action='store_false', help='Do not train word2vec')
	# parser.set_defaults(train_word2vec=True)
	parser.add_argument('-emb1_size', type=int, default=768, help='Embedding dimensions of inputs')
	parser.add_argument('-emb_name', type=str, default='roberta-base', choices=['bert-base-uncased', 'roberta-base'], help='Which pre-trained model')
	# parser.add_argument('-bert_size', type=int, default = 768, help = 'Size of BERT\'s last layer representations')
	parser.add_argument('-emb_lr', type=float, default=1e-5, help='Larning rate to train embeddings')
	parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Freeze embedding weights')
	parser.add_argument('-no-freeze_emb', dest='freeze_emb', action='store_false', help='Train embedding weights')
	parser.set_defaults(freeze_emb=False)

	parser.add_argument('-grade_disp', dest='grade_disp', action='store_true', help='Display grade information in validation outputs')
	parser.add_argument('-no-grade_disp', dest='grade_disp', action='store_false', help='Don\'t display grade information')
	parser.set_defaults(grade_disp=False)
	parser.add_argument('-type_disp', dest='type_disp', action='store_true', help='Display Type information in validation outputs')
	parser.add_argument('-no-type_disp', dest='type_disp', action='store_false', help='Don\'t display Type information')
	parser.set_defaults(type_disp=False)
	parser.add_argument('-challenge_disp', dest='challenge_disp', action='store_true', help='Display information in validation outputs')
	parser.add_argument('-no-challenge_disp', dest='challenge_disp', action='store_false', help='Don\'t display information')
	parser.set_defaults(challenge_disp=False)
	parser.add_argument('-nums_disp', dest='nums_disp', action='store_true', help='Display number of numbers information in validation outputs')
	parser.add_argument('-no-nums_disp', dest='nums_disp', action='store_false', help='Don\'t display number of numbers information')
	parser.set_defaults(nums_disp=True)
	parser.add_argument('-more_nums', dest='more_nums', action='store_true', help='More numbers in Voc2')
	parser.add_argument('-no-more_nums', dest='more_nums', action='store_false', help='Usual numbers in Voc2')
	parser.set_defaults(more_nums=False)
	parser.add_argument('-mawps_vocab', dest='mawps_vocab', action='store_true', help='Custom Numbers in Voc2')
	parser.add_argument('-no-mawps_vocab', dest='mawps_vocab', action='store_false', help='No Custom Numbers in Voc2')
	parser.set_defaults(mawps_vocab=False)

	parser.add_argument('-show_train_acc', dest='show_train_acc', action='store_true', help='Calculate the train accuracy')
	parser.add_argument('-no-show_train_acc', dest='show_train_acc', action='store_false', help='Don\'t calculate the train accuracy')
	parser.set_defaults(show_train_acc=True)

	parser.add_argument('-full_cv', dest='full_cv', action='store_true', help='5-fold CV')
	parser.add_argument('-no-full_cv', dest='full_cv', action='store_false', help='No 5-fold CV')
	parser.set_defaults(full_cv=False)

	#Conf parameters
	parser.add_argument('-conf', type = str, default = 'posterior', choices = ["posterior", "similarity"], help = 'Confidence estimation criteria to use, ["posterior", "similarity"]')
	parser.add_argument('-sim_criteria', type = str, default = 'bleu', choices = ['bert_score', 'bleu_score'], help = 'Only applicable if similarity based criteria is selected for confidence.')
	parser.add_argument('-adv', action = 'store_true', help = 'If dealing with out of distribution examples')
	
	return parser
