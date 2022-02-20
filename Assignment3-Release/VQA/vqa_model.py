import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class Enc(nn.Module):
    def __init__(self, embed_size):
        super(Enc, self).__init__()
        """
        Implement the encoder of images.
        The encoder is like the "linear" mode we implemented in model.py, except that
        we use FC layer to output features of embed_size.
        """
        #########################################################Your Code###########

        #############################################################################

    def forward(self, images):
        features = None
        #########################################Your Code###########################

        #############################################################################
        return features

class Dec(nn.Module):
    def __init__(self, embed_size, ques_vocab_size, ans_vocab_size,  max_seq_length=26,):
        super(Dec, self).__init__()

        self.ques_vocab_size = ques_vocab_size
        self.max_seg_length = max_seq_length

        self.linear = nn.Linear(embed_size + ques_vocab_size, ans_vocab_size)
        

    def forward(self, features, captions):
        outputs = None
        """
        Extract language features
        turn captions into bow features
        specifically, now caption is a list of indices. e.g., [4,5,2,2]
        then bow features should be: [0,0,2,0,1,1]
        """
        #################################Your Code#######################################

        ##################################################################################
        """
        concat image features and language features
        """
        #################################Your Code#######################################

        ##################################################################################
        """
        Input the hidden features into FC layer,
        then output the log softmax probabilities 
        """
        #################################Your Code#######################################

        ##################################################################################
        return outputs

class EncDec(nn.Module):
    def __init__(self, embed_size, 
                       vocab_size, 
                       ans_vocab_size, 
                       max_seq_length=26,
                       prefix_n=1):

        super(EncDec, self).__init__()
        self.embed_size     = embed_size
        self.vocab_size     = vocab_size
        self.ans_vocab_size = ans_vocab_size
        self.max_seq_length = max_seq_length
        self.encoder = Enc(embed_size)
        self.decoder = Dec(embed_size, vocab_size, ans_vocab_size, 
                           max_seq_length)
    
    def forward(self, images, questions):

        """
        first encode images to get image features
        then input image features and questions to the decoder to decode answers
        """
        logits = None
        ##########################################Your Code############################################

        ###############################################################################################
        return logits 
