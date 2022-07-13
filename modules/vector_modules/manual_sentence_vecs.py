#NOTE: USED FOR DEBUG ONLY
from laserembeddings import Laser

class SentenceEmbedder():
    def __init__(self):
        self.laser = Laser()
    
    def _get_laser_embedding(self, sentences, lang='en'):
        '''
        Returns LASER embeddings
        '''
        return self.laser.embed_sentences(sentences,lang=lang)
    
    def get_standard_laser_embedding(self, sentences):
        return self._get_laser_embedding(sentences)
    
    def get_reduced_laser_embedding(self, sentences, new_dimension):
        '''
        Old dimension for laser will be 1024, new will be 128 based on Thompson's
        Expirementation
        '''
        embeddings = self._get_laser_embedding(sentences)
        return dimensionality_reduction(embeddings, new_dimension)