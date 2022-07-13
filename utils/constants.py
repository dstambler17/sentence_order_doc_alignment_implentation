import uuid

class DocumentInfo():
    '''
    Contains Different Representations of the document
    '''
    def __init__(self, doc_vector, url, sentence_embeddings, tokenized_document):
        self.id = uuid.uuid1()
        self.doc_vector = doc_vector
        self.url = url
        self.sentence_embeddings = sentence_embeddings
        self.tokenized_document = tokenized_document
        self.dictified_doc = self._dictify_sentences()
    
    def _dictify_sentences(self):
        '''
        Turns list of sentences into dict that looks something like this
            {"text text text" : 1,
            "hi hi hi": 2,
            "more text": 3}
        '''
        return {sent: i for i, sent in enumerate(self.tokenized_document)}
    
    def get_vec_align_info(self):
        return self.dictified_doc, self.sentence_embeddings, self.tokenized_document

        

class CandidatePair():
    '''
    Contains Two docs in a candidate pair
    '''
    def __init__(self, src_doc: DocumentInfo, target_doc: DocumentInfo, score : float = None):
        self.src_doc = src_doc
        self.target_doc = target_doc
        self.score = score
        self.rescore = None
        self.sentence_alignments = None
    
    def update_score(self, new_score, is_rescore=False):
        if is_rescore:
            self.is_rescore = new_score
        else:
            self.score = new_score
    
    def get_documents(self):
        return self.src_doc, self.target_doc
    
    def get_scores(self):
        return self.score, self.rescore
    
    def get_url_pairs(self):
        return self.src_doc.url, self.target_doc.url
    
    def get_sentence_alignments(self):
        return self.sentence_alignments

    def print_info(self):
        '''Returns a printout of aligned docs and their scores'''
        return NotImplemented 