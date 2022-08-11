import torch
import time
from numpy import dot
from numpy.linalg import norm
from transformers import XLMTokenizer, XLMModel, AutoTokenizer, AutoModel

from laserembeddings import Laser


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(model, tokenizer, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print("Sentence embeddings:")
    print(sentence_embeddings, sentence_embeddings.shape)
    return sentence_embeddings[0].numpy()




tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-100-1280")
model = AutoModel.from_pretrained("xlm-mlm-100-1280")

#NOTE: Why is cosine similarity like this? Is this the lang component bias
#OH I see, it is the lang component

en_sentences = ['The dog jumped over the fence']
ru_sentences = ['Le chien a sauté par-dessus la clôture']

en_embeds = get_embeddings(model, tokenizer, en_sentences)
print('\n Ru Sentences')
ru_embeds = get_embeddings(model, tokenizer, ru_sentences)

#print(en_embeds.shape, ru_embeds.shape)
cos_sim_score = dot(en_embeds, ru_embeds)/(norm(ru_embeds)*norm(en_embeds))
print('cos_sim_score : ', cos_sim_score)


print("\nLASER Embeds for comp")
laser = Laser()

# if all sentences are in the same language:
en_laser_embeddings = laser.embed_sentences(en_sentences, lang='en')
fr_laser_embeddings = laser.embed_sentences(ru_sentences, lang='fr')  

cos_sim_score = dot(en_laser_embeddings, fr_laser_embeddings)/(norm(en_laser_embeddings)*norm(fr_laser_embeddings))
print('cos_sim_score_laser : ', cos_sim_score)
