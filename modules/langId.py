import fasttext.util
import os

PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'
PRETRAINED_MODEL_PATH = '/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/modules/lang_id_models/lid.176.bin'
SUPPORTED_LANGS = """af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br\
                     bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo \
                     es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu \
                     hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez\
                     li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds\
                     ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn\
                     sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec\
                     vep vi vls vo wa war wuu xal xmf yi yo yue zh"""

#If not exist: do this:
# wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
#Use fasttext for this: lid_predictions[0]
#This is fasttext lang id model
class LangIdentification():
    def __init__(self, pretrained_model_path=PRETRAINED_MODEL_PATH, lang_str=SUPPORTED_LANGS):
        fasttext.FastText.eprint = lambda x: None #Suppress warning: https://stackoverflow.com/questions/66353366/cant-suppress-fasttext-warning-load-model-does-not-return
        self.model = fasttext.load_model(PRETRAINED_MODEL_PATH)
        self.lang_codes = [lang for lang in SUPPORTED_LANGS.split(" ") if lang != ""]

    def _clean_input(self, inp):
        return inp.replace('\n', '')

    def _get_language_probabilities_dict(self, raw_input) -> dict:
        """
        Helper function for cleaning model output
        input looks like (('__label__en',), array([0.83386093]))
        or 
         ([['__label__en']], [array([0.83386093], dtype=float32)])
         output looks like
         {'en': 0.83386093}
        """
        langs, probs = raw_input
        sentences_language_dicts = []
        for sent_langs, sent_probs in zip(langs, probs):
            sent_tuples = [(label.split('__label__')[1], prob) for label, prob in zip(sent_langs, sent_probs)]
            sent_dict = {label: prob for label, prob in sent_tuples}
            sentences_language_dicts.append(sent_dict)
        
        return sentences_language_dicts

    def _handle_prediction(self, input : list, K=1):
        '''
        Returns lang_probs after making prediction
        '''
        if type(input) is str:
            input = [input]
        assert type(input) is list
        predictions = self.model.predict([self._clean_input(inp) for inp in input], k=K)
        return  self._get_language_probabilities_dict(predictions)

    def get_language_id(self, input):
        '''
        Function returns lang id, confidence in prediction
        input is either list or str

        Returns tuple
        '''
        lang_id, prediction_confidence = None, None
        for k, v in self._handle_prediction(input)[0].items():
            lang_id, prediction_confidence = k,v
        return lang_id, prediction_confidence
    
    def get_lang_id_probabilities_dict(self, input):
        return self._handle_prediction(input, K = len(self.lang_codes))


if __name__ == "__main__":
    lid = LangIdentification()
    sentences = ['ආයුබෝවන්', 'ඔබට කෙසේද']
    lid_predictions = lid.get_lang_id_probabilities_dict(sentences)
    lid_predictions[0].get('si', 0)

    import pdb
    pdb.set_trace()
