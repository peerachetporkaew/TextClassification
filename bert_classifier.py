from transformers import AutoTokenizer, AutoModel
from attacut import tokenize
import torch.nn as nn
import torch

tokenizer = AutoTokenizer.from_pretrained("lst-nectec/HoogBERTa")


class HoogBERTaClassifier(nn.Module):

    def __init__(self,n_classes = 2):

        super().__init__()
        self.bert = AutoModel.from_pretrained("lst-nectec/HoogBERTa")
        self.linear = nn.Linear(768,2)


    def forward(self,tokenized_text): # tokenized_text : huggingface tokenized_text 
        features = self.bert(**tokenized_text, output_hidden_states = True).hidden_states[-1] # (batch, seq_dim, hidden dim)

        input_features = features[:,0,:] # (batch, hidden dim)

        logits = self.linear(input_features)
        return logits


def my_tokenizer(sentence, return_tensor=True):
    all_sent = []
    sentences = sentence.split(" ")
    for sent in sentences:
        all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))

    sentence = " _ ".join(all_sent)
    if return_tensor:
        tokenized_text = tokenizer(sentence, return_tensors = 'pt')
        return tokenized_text
    
    else:
        # token_ids = tokenized_text['input_ids']
        return sentence

def my_batch_tokenizer(sentenceL):
    inputList = []
    for sentence in sentenceL:
        inputList.append(my_tokenizer(sentence,return_tensor=False))
    
    # token_ids = tokenized_text['input_ids']
    tokenized_text = tokenizer(inputList, return_tensors = 'pt',padding=True)
    return tokenized_text