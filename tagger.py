
# coding: utf-8

# In[1]:


import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


def prepare_sequence(seq, to_ix):
    """ Converts the sequence into list of indices mapped by to_ix.
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# In[3]:


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
       
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


    def report_accuracy(self, data, word_to_ix, tag_to_ix, ix_to_tag, print_data=False):
        """ Reports accuracy with respect to exact match (all tags correct per sentence)
            and total matches (all correctly classified tags).
        """
        # Here we don't need to train, so the code is wrapped in torch.no_grad() 
        with torch.no_grad(): 
            total = 0 
            total_correct = 0 
            total_exact_correct = 0 
            for sentence, tags in data: 
                scores = self(prepare_sequence(sentence, word_to_ix)) 
                out = torch.argmax(scores, dim=1) 
                out_tags = [ix_to_tag[ix] for ix in out] 
                targets = prepare_sequence(tags, tag_to_ix) 
     
                correct = 0 
                length = len(tags) 
                for i in range(length): 
                    if out[i] == targets[i]: 
                        correct += 1 

                total += length
                total_correct += correct


            n = len(data)
           
            print('Accuracy: %d / %d, %0.4f' % (total_correct, total, total_correct / total))


# In[4]:


# Our model and helper functions.
from lstm import LSTMTagger, prepare_sequence

torch.manual_seed(1)

# These will usually be 32 or 64 dimensional (little sense to go above 100).
EMBEDDING_DIM = 64

HIDDEN_DIM = 10

# There will usually be more epochs; use 5 or lower to debug.
EPOCHS = 5

TAGS = ["", "#", "$", "''", "(", ")", ",", ".", ":", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "``"]

def main(train_file):
    # Load the data.
    training_data = read_data(train_file)
    n = len(training_data)
    ########
    #print(n)

    # Store word -> word_index mapping.
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    ##print(word_to_ix)

    # Store tag -> tag_index mapping.
    tag_to_ix = {tag: ix for ix, tag in enumerate(TAGS)}
    ##################
    #print(tag_to_ix)
    ##################
    
    # Initialize the model.
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(EPOCHS):
        for i, (sentence, tags) in enumerate(training_data):
            # Step 1. Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
               
                print('Epoch %d, sentence %d/%d, loss: %0.4f' % (epoch + 1, i + 1, n, loss))
                
    
                

    # Report training accuracy
    model.report_accuracy(training_data, word_to_ix, tag_to_ix, TAGS)

    


# In[7]:


from sklearn.model_selection import train_test_split

def read_data(filename):
    """ Reads a vertical corpus with two columns: word, pos-tag.
        Returns: list of tuples: [(words, tags)], one record per sentence.
    """
    data = []

    with open("data/train.txt") as datafile:
        words = []
        tags = []
        for line in datafile:
            line = line.rstrip()
            if not line:
                data.append((words, tags))
                words = []
                tags = []
            else:
                word, tag = line.split()
                words.append(word)
                tags.append(tag)
    
    
            
    return data
  


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: tagger.py TRAINSET', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

