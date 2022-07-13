import math

class BoilerplateDownWeighting():
    '''
    Parent class for all downweighting schemes

    name: name of the boiler plate down weighting technique
    '''
    def __init__(self, name):
        self.description = description
    
    def get_weighting_score(self, sentence):
        return NotImplemented

class InvserseDownWeightingWithLog(BoilerplateDownWeighting):
    '''
    Looks Something like this:
    #1/(1+log(count))
    #1/(1+log(3000))
    '''
    def __init__(self, documents):
        self.documents = [{sent: True for sent in doc} for doc in documents]
        self.description ="inverse of the log of number of docs with given sentence"
    
    def get_weighting_score(self, sentence):
        apperances = 0
        for doc in self.documents:
            if sentence in doc:
                apperances += 1
        return 1/(1+ math.log(apperances))

class LIDFDownWeighting(BoilerplateDownWeighting):
    def __init__(self, documents):
        self.documents = [{sent: True for sent in doc} for doc in documents]
        self.description ="h scales sentences by the inverse of the \
                          (linear, as opposed to log) number of documents \
                          containing a given sentence"
    def get_weighting_score(self, sentence):
        apperances = 0
        for doc in self.documents:
            if sentence in doc:
                apperances += 1
        assert apperances != 0
        return 1/(max(apperances, 1)) #Add max of 1 for sanity checking

class LengthDownWeighting(BoilerplateDownWeighting):
    def __init__(self):
        self.description ="scales by length in chars of sentence"

    def get_weighting_score(self, sentence):
        return len(sentence)

#NOTE: Paper does not mention this 
class IDFDownWeight(BoilerplateDownWeighting):
    '''
    Implements downweighting with the IDF formula
    '''
    def __init__(self, documents):
        self.documents = [{sent: True for sent in doc.split('.')} for doc in documents]
        self.description ="Scales using the IDF Formula"
        
    def get_weighting_score(self, sentence):
        apperances = 0
        for doc in self.documents:
            if sentence in doc:
                apperances += 1
        num_docs = len(self.documents)
        assert apperances != 0
        return math.log(num_docs/(max(apperances, 1))) #Add max of 1 for sanity checking
