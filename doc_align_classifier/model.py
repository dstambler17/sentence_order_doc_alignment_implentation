import torch
import torch.nn as nn


class DocAlignerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size_one=256, hidden_size_two=128, hidden_size_three=64):
        super(DocAlignerClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size_two)
        self.fc2 = nn.Linear(hidden_size_two, hidden_size_three)
        #self.fc3 = nn.Linear( hidden_size_two, hidden_size_three) #TEMP: MAYBE ADD LATER
        self.fc_out = nn.Linear(hidden_size_three, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #Optional
        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
        nn.init.kaiming_uniform_(self.fc1.weight) #He init for relu activation layers
        nn.init.kaiming_uniform_(self.fc2.weight)
        #nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc_out.weight) #Xavier init for sigmoid

                
    def forward(self, src_doc_embed, tgt_doc_embed):
        '''
        Forward data through the lstm
        '''
        combined_input = torch.cat((src_doc_embed.view(src_doc_embed.size(0), -1),
                          tgt_doc_embed.view(tgt_doc_embed.size(0), -1)), dim=1)
        #print(combined_input.shape, "combed input")

        x = self.relu(self.batchnorm1(self.fc1(combined_input)))
        x = self.relu(self.batchnorm2(self.fc2(x)))
        #x = self.relu(self.fc3(x)) 
        #x = self.batchnorm2(x) #NOTE: Better to leave out batch norm
        x = self.dropout(x) #NOTE: Use this when using three layers to prevent overfitting
        output = self.sigmoid(self.fc_out(x))

        return output


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    GithubCode: https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_sent_vecs):
        """
        input:
            batch_sent_vecs : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_sent_vecs).squeeze(-1)).unsqueeze(-1)
        pooled_doc_rep = torch.sum(batch_sent_vecs * att_w, dim=1)

        return pooled_doc_rep

class SentenceInputAttnClassifier(nn.Module):
    def __init__(self, input_size, hidden_size_one=256, hidden_size_two=128, hidden_size_three=64):
        '''pass'''
        super(SentenceInputAttnClassifier, self).__init__()
        self.attention_layer = SelfAttentionPooling(1024) #TODO: reduce doc vector dim to 64 instead of 128 to match sent embed size
        self.fc1 = nn.Linear(input_size, hidden_size_two)
        self.fc2 = nn.Linear(hidden_size_two, hidden_size_three)
        #self.fc3 = nn.Linear( hidden_size_two, hidden_size_three) #TEMP: MAYBE ADD LATER
        self.fc_out = nn.Linear(hidden_size_three, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #Optional
        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
        nn.init.kaiming_uniform_(self.fc1.weight) #He init for relu activation layers
        nn.init.kaiming_uniform_(self.fc2.weight)
        #nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc_out.weight) #Xavier init for sigmoid

    
    def forward(self, src_embed_vecs, src_doc_vec, tgt_embed_vecs, tgt_doc_vec):
        '''input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        '''
        
        # First get the learned pooled representation for src and tgt docs from LASER sents
        # Then concat
        src_doc_repr = self.attention_layer(torch.cat(src_embed_vecs, src_doc_vec))
        tgt_doc_repr = self.attention_layer(torch.cat(tgt_embed_vecs, tgt_doc_vec))
        combined_input = torch.cat((src_doc_repr.view(src_doc_repr.size(0), -1),
                    tgt_doc_repr.view(tgt_doc_repr.size(0), -1)), dim=1)

        x = self.relu(self.batchnorm1(self.fc1(combined_input)))
        x = self.relu(self.batchnorm2(self.fc2(x)))
        #x = self.relu(self.fc3(x)) 
        #x = self.batchnorm2(x) #NOTE: Better to leave out batch norm
        x = self.dropout(x) #NOTE: Use this when using three layers to prevent overfitting
        output = self.sigmoid(self.fc_out(x))

