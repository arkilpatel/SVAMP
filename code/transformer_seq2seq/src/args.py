import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')

	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	
	# Run Config
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='asdiv-a_fold0_final', help='Dataset')
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
	parser.set_defaults(histogram=False)
	parser.add_argument('-save_writer', dest='save_writer',action='store_true', help='To write tensorboard')
	parser.add_argument('-no-save_writer', dest='save_writer', action='store_false', help='Dont write tensorboard')
	parser.set_defaults(save_writer=False)

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=2, help='Specify the gpu to use')
	parser.add_argument('-early_stopping', type=int, default=500, help='Early Stopping after n epoch')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=False)

	# Transformer parameters
	parser.add_argument('-heads', type=int, default=8, help='Number of Attention Heads')
	parser.add_argument('-encoder_layers', type=int, default=6, help='Number of layers in encoder')
	parser.add_argument('-decoder_layers', type=int, default=6, help='Number of layers in decoder')
	parser.add_argument('-d_model', type=int, default=300, help='the number of expected features in the encoder inputs') #768? features of BERT? HAS TO BE 300 if using word2Vec
	parser.add_argument('-d_ff', type=int, default=1200, help='Embedding dimensions of intermediate FFN Layer (refer Vaswani et. al)')
	parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('-dropout', type=float, default=0.1, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	parser.add_argument('-warmup', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for')
	parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=16, help='Batch size')

	parser.add_argument('-max_length', type=int, default=80, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	
	parser.add_argument('-embedding', type=str, default='word2vec', choices=['bert', 'roberta', 'word2vec', 'random'], help='Embeddings')
	parser.add_argument('-word2vec_bin', type=str, default='/datadrive/satwik/global_data/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')
	parser.add_argument('-emb_name', type=str, default='roberta-base', choices=['bert-base-uncased', 'roberta-base'], help='Which pre-trained model')
	parser.add_argument('-emb_lr', type=float, default=1e-5, help='Larning rate to train embeddings')
	parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Freeze embedding weights')
	parser.add_argument('-no-freeze_emb', dest='freeze_emb', action='store_false', help='Train embedding weights')
	parser.set_defaults(freeze_emb=False)

	parser.add_argument('-epochs', type=int, default=10, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='adamw', choices=['adam', 'adamw', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')

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
	parser.set_defaults(show_train_acc=False)

	parser.add_argument('-full_cv', dest='full_cv', action='store_true', help='5-fold CV')
	parser.add_argument('-no-full_cv', dest='full_cv', action='store_false', help='No 5-fold CV')
	parser.set_defaults(full_cv=False)

	return parser
