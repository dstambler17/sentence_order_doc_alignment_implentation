import faiss
from utils.constants import CandidatePair
from utils.common import function_timer

class ApproximateNearestNeighborSearch():
    '''
    Uses Faiss #Paper uses FAISS approximate (compressed-domain) nearest neighbor search implementation
    # https://github.com/facebookresearch/faiss
    '''
    def __init__(self, dim_size):
        #self.index = faiss.IndexFlatL2(dim_size)
        self.index = faiss.index_factory(dim_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        #self.index = faiss.IndexFlatIP(d) NOTE: This looks to be same as...


    def _get_K_nearest_neighbors(self, src_lang_matrix, target_lang_matrix, K, DEBUG_MODE=False):
        '''
        Debug Mode = if Distances should be returned
        '''
        src_lang_matrix = src_lang_matrix.astype('float32')
        target_lang_matrix = target_lang_matrix.astype('float32')

        #self.index.ntotal
        #faiss.normalize_L2(target_lang_matrix)

        self.index.add(target_lang_matrix)
        #faiss.normalize_L2(src_lang_matrix)
        D, I = self.index.search(src_lang_matrix, K)
        return I, D
        
    @function_timer
    def build_candidate_pair_list(self, src_docs, tgt_docs, src_matrix, tgt_matrix, K_VAL=10, IS_DEBUG=False):
        '''
        Builds a list of candidate pairs
        src_docs: list of Document Objects for src lang
        trg_docs: list of Document Objects for trg lang
        src_matrix: matrix containing src embedding info, used to build KNN model
        trg_matrix: matrix containing trg embedding info, used to eval KNN model

        returns a list of candidate pairs
        '''
        
        K = min(len(tgt_matrix), K_VAL)
        print("USING K VAL of %d" % K)
        nearest_neighbors, distances = self._get_K_nearest_neighbors(src_matrix, tgt_matrix, K)
        candidate_pairs = []
        
        if IS_DEBUG:
            debug_pairs = [] #DEBUG ONLY
        for ii, row in enumerate(nearest_neighbors):
            src_doc = src_docs[ii]
            for jj, tgt_idx in enumerate(row):
                tgt_doc = tgt_docs[tgt_idx]
                score = distances[ii][jj]
                candidate_pairs.append(CandidatePair(src_doc, tgt_doc, score))
                if IS_DEBUG:
                    debug_pairs.append((ii, tgt_idx, score))

        return candidate_pairs

@function_timer
def competivieMatchingAlgorithm(candidate_pairs, use_rescore=True):
    '''
    Competitive Matching Algorithm from Buck Koehn Paper
    input list of candidate pair object
    '''
    #Higher the rescore the better
    #Normal score uses cosine dist (1 - cosine sim)
    #However, Approx KNN uses cosine sim, which is why we need to reverse
    if use_rescore:
        candidate_pairs.sort(key=lambda x: x.rescore, reverse=True)
    else:
        candidate_pairs.sort(key=lambda x: x.score, reverse=True)
    aligned = []
    S_one, S_two = {}, {}
    for cand_pair in candidate_pairs:
        doc_one, doc_two = cand_pair.get_documents()
        if doc_one.id not in S_one and doc_two.id not in S_two:
            aligned.append(cand_pair)
            S_one[doc_one.id] = doc_one
            S_two[doc_two.id] = doc_two
    return aligned
    
    


if __name__ == "__main__":
    #TODO: Figure out how to use cosine similarity in this codebase
    #NOTE: I is the indexs that match
    #NOTE: D is the distances
    import numpy as np
    d = 64                           # dimension
    nb = 100000                      # database size
    nq = 10000                       # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.


    import faiss                   # make faiss available
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(xb)  # add vectors to the index
    print(index.ntotal)

    k = 4                    
    D, I = index.search(xq, k)     # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    print(D[:5])                  # neighbors of the 5 last queries
    import pdb
    pdb.set_trace()
