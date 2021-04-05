import numpy as np
import torch
from src.utils.bleu import compute_bleu

def posterior_based_conf(test_ques, model):

    decoded_words, decoded_log_probs = model.greedy_decode(test_ques, return_probs = True)
    posteriors = [np.exp(sum(log_probs)) for log_probs in decoded_log_probs]
    return decoded_words, posteriors

def similarity_based_conf(test_ques, train_ques,model, sim_criteria = 'bert_score'):
    '''
    Takes a batch of test question and evaluates their closest similarities between questions in training set.
    Inputs:
        test_ques: A list of strings containing a batch of test questions. Length: Batch Size
        train_ques: A list containing **ALL** the questions present in training data. Length: |Training Data|
        model: bert_seq2exp model
        sim_criteria: Criteria used to evaluate similarity between test questions and training questions

    Returns a numpy array containing closest similarity of each test input in the batch size. Shape: [Batch Size,]
    '''

    decoded_words = model.greedy_decode(test_ques)
    if sim_criteria == 'bert_score':
        similarities = bert_sim(test_ques, train_ques, model) #[Batch Size x |Training Data|]

    elif sim_criteria == 'bleu_score':
        similarities = bleu_sim(test_ques, train_ques)
    else:
        raise ValueError("Other similarity methods not implemented yet!")

    max_sims = np.max(similarities, axis = 1)
    return decoded_words, max_sims



def bert_sim(queries, keys, model):
    '''
    Inputs
        - queries: a batch of sentences whose similarity is to be measured with other sentences. Length: L_Q
        - keys: those other sentences. Length: L_K
        - model: bert_seq2exp model

    Outputs: A numpy array containing similarites between each test sentence with all training examples. Shape: [L_Q, L_K]
    '''

    #Feed queries and keys to bert and obtain contextualized representation, using embeddings of [CLS]
    #  (TODO: try pooling instead of [CLS])
    with torch.no_grad():
        queries_rep     = model.bert(queries)[0][:,0].detach().cpu().numpy()
        keys_rep        = torch.cat([model.bert(keys[i:min(i+16, len(keys)),])[0][:,0] for i in range(0, len(keys), 16)], dim = 0)
        keys_rep        = keys_rep.detach().cpu().numpy()

    sims = np.dot(queries_rep / np.linalg.norm(queries_rep, axis = -1, keepdims = True),
                 (keys_rep / np.linalg.norm(keys_rep, axis = -1, keepdims = True)).T)
    return sims


def bleu_sim(queries, keys):
    '''
    Inputs:
        - queries: a batch of sentences whose similarity is to be measured with other sentences. Length: L_Q
        - keys: those other sentences. Length: L_K

    Outputs: A numpy array containing bleu scores between each test sentence with all training examples. Shape: [L_Q, L_K]
    '''
    bleus = [[] for i in range(len(queries))]
    for i in range(len(queries)):
        for j in range(len(keys)):
            refs = [[keys[j].split()]]
            hyps = [queries[i].split()]
            bleu = compute_bleu(refs, hyps)[0]
            bleus[i].append(bleu)

    sims = np.array(bleus)
    return sims
