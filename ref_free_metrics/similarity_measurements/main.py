import sys
import os
import copy
sys.path.append('../..')
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from resources import BASE_DIR, BERT_TYPE_PATH_DIC, SENT_TRANSFORMER_TYPE_PATH_DIC
from my_sentence_transformers import SentenceTransformer
from ref_free_metrics.similarity_scorer import parse_documents
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_token_vecs


def mrg_tokens_sents(tokens_vecs, token_weights, sents_vecs, sents_weights):
    mrgd_vecs = []
    mrgd_weights = []
    for tvec, tweights, svec, sweights in zip(tokens_vecs, token_weights, sents_vecs, sents_weights):
        if svec is not None and tvec is not None:
            mrgd_vecs.append(np.concatenate([tvec, svec], axis=0))  # (nstpw_num + sent_num, dim)
            mrgd_weights.append(np.concatenate([tweights, sweights]))
        elif svec is not None:
            # if tvec is None, sve may not be None
            mrgd_vecs.append(svec)  # (sent_num, dim)
            mrgd_weights.append(sweights)
        elif tvec is not None:
            # if svec is None, tve may not be None
            mrgd_vecs.append(tvec)  # (nstpw_num, dim)
            mrgd_weights.append(tweights)
        else:
            mrgd_vecs.append(None)
            mrgd_weights.append(None)
    return mrgd_vecs, mrgd_weights

def get_idf(doc_token_list):
    df_dic = {}
    for i,doc_tokens in enumerate(doc_token_list):
        if doc_tokens is None: continue
        for tk in doc_tokens:
            if tk in df_dic: df_dic[tk].append(i)
            else: df_dic[tk] = [i]

    doc_num = len(doc_token_list)
    idf_list = []
    for i,doc_tokens in enumerate(doc_token_list):
        if doc_tokens is None:
            idf_list.append(None)
            continue
        idf = []
        for tk in doc_tokens: idf.append(-1.*np.log( (len(set(df_dic[tk]))+0.5)/(doc_num+0.5)))
        idf_list.append(np.array(idf))
    return idf_list

def get_my_score(ref_vecs, ref_weights, ref_tokens, summ_vecs, summ_weights, summ_tokens, wmd_score_type, wmd_weight_type, mask_self=False, beta_gamma=2):
    recall_list = []
    precision_list = []
    f1_list = []
    empty_summs_ids = []

    if mask_self:
        assert wmd_weight_type == 'none'
        assert wmd_score_type == 'recall'

    if 'idf' in wmd_weight_type:
        final_ref_weights = get_idf(ref_tokens)
        final_summ_weights = get_idf(summ_tokens)
    elif 'graph_weighted' in wmd_weight_type:
        final_ref_weights = ref_weights
        final_summ_weights = summ_weights

    if 'renormalize' in wmd_weight_type:
        final_ref_weights = [final_ref_weights[i] / final_ref_weights[i].sum() if final_ref_weights[i] is not None else None for i in range(len(final_ref_weights))]
        final_summ_weights = [final_summ_weights[i] / final_summ_weights[i].sum() if final_summ_weights[i] is not None else None for i in range(len(final_summ_weights))]

    for i,rvecs in enumerate(ref_vecs):
        r_recall_list = []
        r_precision_list = []
        r_f1_list = []
        for j,svecs in enumerate(summ_vecs):
            if svecs is None or len(svecs) == 0:
                empty_summs_ids.append(j)
                r_recall_list.append(None)
                r_precision_list.append(None)
                r_f1_list.append(None)
                continue
            if mask_self:
                # only token level information is utilized
                assert rvecs.shape[0] == len(ref_tokens[i])
                assert svecs.shape[0] == len(summ_tokens[j])
                # the matrix should be square matrix
                assert rvecs.shape[0] == svecs.shape[0]
            sim_matrix = cosine_similarity(rvecs,svecs)
            if mask_self:
                np.fill_diagonal(sim_matrix, 0)
            beta_square = 1
            if wmd_score_type == 'f1_beta':
                beta_square = (rvecs.shape[0] / svecs.shape[0]) ** (1/beta_gamma)
                beta_square = 2 if beta_square > 2 else beta_square
                beta_square = 1 if beta_square < 1 else beta_square
            if 'idf' in wmd_weight_type or 'graph_weighted' in wmd_weight_type:
                weighted_recall = np.dot(np.max(sim_matrix, axis=1), final_ref_weights[i])
                weighted_precision = np.dot(np.max(sim_matrix, axis=0), final_summ_weights[j])
                weighted_f1 = (1. + beta_square) * weighted_recall * weighted_precision / (weighted_recall + beta_square * weighted_precision)
                r_recall_list.append(weighted_recall)
                r_precision_list.append(weighted_precision)
                r_f1_list.append(weighted_f1)
            else:
                recall = np.mean(np.max(sim_matrix, axis=1))
                precision = np.mean(np.max(sim_matrix, axis=0))
                f1 = (1. + beta_square) * recall * precision / (recall + beta_square * precision)
                r_recall_list.append(recall)
                r_precision_list.append(precision)
                r_f1_list.append(f1)
        recall_list.append(r_recall_list)
        precision_list.append(r_precision_list)
        f1_list.append(r_f1_list)
    empty_summs_ids = list(set(empty_summs_ids))
    recall_list = np.array(recall_list)
    precision_list = np.array(precision_list)
    f1_list = np.array(f1_list)
    if 'recall' in wmd_score_type:
        scores = []
        for i in range(len(summ_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(recall_list[:,i]))
        return scores
        #return np.mean(np.array(recall_list), axis=0)
    elif 'precision' in wmd_score_type:
        scores = []
        for i in range(len(summ_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(precision_list[:,i]))
        return scores
        #return np.mean(np.array(precision_list), axis=0)
    else:
        assert 'f1' in wmd_score_type
        scores = []
        for i in range(len(summ_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(f1_list[:,i]))
        return scores
        #return np.mean(np.mean(f1_list),axis=0)


def tfrf(documents, summaries, model_path):
    """
    return a list of scores for given summaries using a model at model_path 
    """
    ref_metric = 'top12'
    map_type='t2t'
    sent_represnt_type='mean_all'
    ref_st_mrg_type='wAll_sAll'
    lambda_redund=0.0
    pacsum_beta=0.0
    pacsum_lambda1=2.0
    pacsum_lambda2=1.0
    beta_gamma=2
    doc_num_limit = -1
    wmd_score_type ='f1' #choices=['f1', 'precision', 'recall', 'f1_beta']
    wmd_weight_type = 'idf_renormalize' #choices=['global_idf_renormalize', 'idf_renormalize', 'none', 'graph_weighted_renormalize']


    corpus_read = CorpusReader(BASE_DIR)
    # assume all documents about one topic
    docs = []
    for doc in documents:
        entry = corpus_read.readOneDoc(doc)
        docs.append((doc,entry))
    
    peer_summaries = PeerSummaryReader(BASE_DIR)
    # assume all summeris are about one topic
    summs = []
    for peer in summaries:
        sents = peer_summaries.readOnePeer(peer)
        summs.append((peer, sents))

    models = corpus_read.readModels(model_path)
    sent_transformer_type='bert_large_nli_stsb_mean_tokens'
    sent_transformer_path = SENT_TRANSFORMER_TYPE_PATH_DIC[sent_transformer_type]
    bert_model = SentenceTransformer(sent_transformer_path, device='cpu')


    # for topic,docs,models in corpus_reader(year):
    # select docs
    if doc_num_limit > 0:
        docs = docs[:doc_num_limit]
    
    # print('\n=====Topic{}: {}====='.format(topic_idx, topic))
    sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(docs, bert_model, ref_metric,sent_represnt_type,
                                                                                        pacsum_beta=pacsum_beta, pacsum_lambda1=pacsum_lambda1,
                                                                                        pacsum_lambda2=pacsum_lambda2)
    
    ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k] > 0.0} # wchen: '>=0.1' -> '> 0.0'
    ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
    ref_sources = sorted(list(ref_sources))
    # get sents in ref/doc
    ref_sents = []
    ref_sents_vecs = []
    ref_sents_weights = []
    ref_tokens_vecs = []
    ref_tokens = []
    sorted_ref_dic_keys = sorted(ref_dic.keys())

    for rs in ref_sources:
        ref_sents.append([ref_dic[k]['text'] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_sents_vecs.append([sent_vecs[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_sents_weights.append([sents_weights[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_tokens_vecs.append([token_vecs[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_tokens.append([all_tokens[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])

    # get the filtered vecs for ref
    filtered_ref_tokens_vecs = []
    filtered_ref_tokens = []
    filtered_ref_token_weights = []

    if ref_st_mrg_type.startswith('wTop'):
        wTopNum = int(ref_st_mrg_type.split('_')[0].strip()[4:])
        token_level_idx_list = []
        sent_level_idx_list = []
        for doc_weights in ref_sents_weights:
            doc_weights = np.array(doc_weights)
            if doc_weights.all():
                idxs_list = [k for k in range(len(doc_weights))]
            else:
                idxs_list = doc_weights.argsort().tolist()
                idxs_list = idxs_list[::-1]
            token_level_idx_list.append(idxs_list[:wTopNum])
            if ref_st_mrg_type.endswith('sBottom'):
                sent_level_idx_list.append(idxs_list[wTopNum:])
            else:
                assert ref_st_mrg_type.endswith('sAll')
                sent_level_idx_list.append(idxs_list)
    else:
        token_level_idx_list = [[k for k in range(len(doc_weights))] for doc_weights in ref_sents_weights]
        sent_level_idx_list = token_level_idx_list

    for doc_idx in range(len(ref_sents)):
        tvecs_in = [ref_tokens_vecs[doc_idx][k] for k in token_level_idx_list[doc_idx]]
        tokens_in = [ref_tokens[doc_idx][k] for k in token_level_idx_list[doc_idx]]
        # we use sent weight as the weight of each token
        weights_in = [np.array([ref_sents_weights[doc_idx][k]]*len(ref_tokens[doc_idx][k])) for k in token_level_idx_list[doc_idx]]
        vv, tt, ww = get_token_vecs(vecs=tvecs_in, tokens=tokens_in, weights=weights_in)
        filtered_ref_tokens_vecs.append(vv)
        filtered_ref_tokens.append(tt)
        filtered_ref_token_weights.append(ww)

    filtered_ref_sents_vecs = []
    filtered_ref_sent_weights = []
    for svec_list, sweights, sent_level_idxs in zip(ref_sents_vecs, ref_sents_weights, sent_level_idx_list):
        remain_svecs = None
        remain_sweights = None
        if len(sent_level_idxs) > 0:
            remain_svecs = [svec_list[k] for k in sent_level_idxs if svec_list[k] is not None]
            remain_sweights = [sweights[k] for k in sent_level_idxs if svec_list[k] is not None]
            if len(remain_svecs) > 0:
                remain_svecs = np.stack(remain_svecs)
                remain_sweights = np.array(remain_sweights)
            else:
                remain_svecs = None
                remain_sweights = None
        filtered_ref_sents_vecs.append(remain_svecs)
        filtered_ref_sent_weights.append(remain_sweights)

    # get sents in system summaries
    filtered_summ_tokens_vecs = []
    filtered_summ_tokens = []
    filtered_summ_token_weights = []
    filtered_summ_sents_vecs = []
    filtered_summ_sent_weights = []

    for ss_idx, ss in enumerate(summs):
        if len(ss[1]) != 0:
            one_summ_sents_vecs, one_summ_tokens_vecs, one_summ_tokens = bert_model.encode(ss[1], sent_represnt_type)
            # print('summary length: {}'.format(sum([len(one_ss_sent) for one_ss_sent in one_summ_tokens]))) # for debug
            vv, tt, _ = get_token_vecs(vecs=one_summ_tokens_vecs, tokens=one_summ_tokens)
            svv = np.stack([svec for svec in one_summ_sents_vecs if svec is not None])
            tweights = np.ones(tt.shape[0])
            sweights = np.ones(svv.shape[0])
        else:
            svv, vv, tt, tweights, sweights = None, None, None, None, None
        filtered_summ_tokens_vecs.append(vv)
        filtered_summ_tokens.append(tt)
        filtered_summ_token_weights.append(tweights)
        filtered_summ_sents_vecs.append(svv)
        filtered_summ_sent_weights.append(sweights)

    # get the merged sent and token representations of references
    filtered_ref_mrgd_vecs, filtered_ref_mrgd_weights = mrg_tokens_sents(filtered_ref_tokens_vecs,
                                                                            filtered_ref_token_weights,
                                                                            filtered_ref_sents_vecs,
                                                                            filtered_ref_sent_weights)
    # get the merged sent and token representations of summs
    filtered_summ_mrgd_vecs, filtered_summ_mrgd_weights = mrg_tokens_sents(filtered_summ_tokens_vecs,
                                                                            filtered_summ_token_weights,
                                                                            filtered_summ_sents_vecs,
                                                                            filtered_summ_sent_weights)
    # get the final input vectors
    assert '2' in map_type
    map_type_ref, map_type_summ = map_type.split('2')
    map_type_ref = map_type_ref.strip()
    map_type_summ = map_type_summ.strip()
    # for ref
    if map_type_ref == 't':
        # token2* mapping
        assert ref_st_mrg_type == 'wAll_sAll'
        final_ref_vecs = filtered_ref_tokens_vecs
        final_ref_weights = filtered_ref_token_weights
    elif map_type_ref == 's':
        # sent2* mapping
        assert 'idf' not in wmd_score_type
        final_ref_vecs = filtered_ref_sents_vecs
        final_ref_weights = filtered_ref_sent_weights
    else:
        # (sent+token)2* mapping
        assert 'idf' not in wmd_score_type
        assert map_type_ref == 'st'
        final_ref_vecs = filtered_ref_mrgd_vecs
        final_ref_weights = filtered_ref_mrgd_weights

    # for summ
    if map_type_summ == 't':
        # *2token mapping
        final_summ_vecs = filtered_summ_tokens_vecs
        final_summ_weights = filtered_summ_token_weights
    elif map_type_summ == 's':
        # *2sent mapping
        assert 'idf' not in wmd_score_type
        final_summ_vecs = filtered_summ_sents_vecs
        final_summ_weights = filtered_summ_sent_weights
    else:
        # *2(sent+token) mapping
        assert 'idf' not in wmd_score_type
        assert map_type_summ == 'st'
        final_summ_vecs = filtered_summ_mrgd_vecs
        final_summ_weights = filtered_summ_mrgd_weights


    # relevance/informativeness score
    relevance_score = get_my_score(final_ref_vecs, final_ref_weights, filtered_ref_tokens,
                                final_summ_vecs, final_summ_weights, filtered_summ_tokens,
                                wmd_score_type, wmd_weight_type, beta_gamma=beta_gamma)
    # redundancy score
    redund_score = []
    for i in range(len(filtered_summ_tokens_vecs)):
        redund_score_i = get_my_score([filtered_summ_tokens_vecs[i]], [filtered_summ_token_weights[i]], [filtered_summ_tokens[i]],
                                    [filtered_summ_tokens_vecs[i]], [filtered_summ_token_weights[i]], [filtered_summ_tokens[i]],
                                    wmd_score_type='recall', wmd_weight_type='none', mask_self=True)
        redund_score.append(redund_score_i[0])
    assert len(relevance_score) == len(redund_score)

    # final score
    pss = []
    for i in range(len(relevance_score)):
        if relevance_score[i] is not None and redund_score[i] is not None:
            pss.append((relevance_score[i] - lambda_redund * redund_score[i]) / (1 + lambda_redund))
        else:
            assert relevance_score[i] is None and redund_score[i] is None
            pss.append(None)

    return pss
