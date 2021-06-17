import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embeddingSize, numberHeads):
        super(SelfAttention, self).__init__()
        self.embeddingSize = embeddingSize
        self.numberHeads = numberHeads
        self.dimsHead = embeddingSize // numberHeads

        if self.dimsHead * numberHeads != embeddingSize:
            print("Hyperparameter error")
            exit(0)

        self.values = nn.Linear(self.dimsHead, self.dimsHead, bias=False)
        self.keys = nn.Linear(self.dimsHead, self.dimsHead, bias=False)
        self.queries = nn.Linear(self.dimsHead, self.dimsHead, bias=False)
        self.fullyConnected = nn.Linear(
            numberHeads * self.dimsHead, embeddingSize)

    def forward(self, values, keys, query, mask):
        totalLength = query.shape[0]
        lenValues, lenKeys, lenQueries = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(totalLength, lenValues, self.numberHeads, self.dimsHead)
        keys = keys.reshape(totalLength, lenKeys, self.numberHeads, self.dimsHead)
        query = query.reshape(totalLength, lenQueries, self.numberHeads, self.dimsHead)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embeddingSize ** (1 / 2)), dim=3)
        output = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(totalLength, lenQueries, self.numberHeads * self.dimsHead)
        output = self.fullyConnected(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embeddingSize, numberHeads, dropout, forwardExp):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embeddingSize, numberHeads)
        self.norm1 = nn.LayerNorm(embeddingSize)
        self.norm2 = nn.LayerNorm(embeddingSize)
        self.feedForward = nn.Sequential(nn.Linear(embeddingSize, forwardExp * embeddingSize),nn.ReLU(),nn.Linear(forwardExp * embeddingSize, embeddingSize),)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feedForward(x)
        output = self.dropout(self.norm2(forward + x))
        return output


class Encoder(nn.Module):
    def __init__(self,sourceVocabSize,embeddingSize,numberLayers,numberHeads,device,forwardExp,dropout,maxLength,):
        super(Encoder, self).__init__()
        self.embeddingSize = embeddingSize
        self.device = device
        self.wordEmbedding = nn.Embedding(sourceVocabSize, embeddingSize)
        self.positionalEmbedding = nn.Embedding(maxLength, embeddingSize)
        self.layers = nn.ModuleList([TransformerBlock(embeddingSize,numberHeads,dropout=dropout,forwardExp=forwardExp,) for _ in range(numberLayers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        totalLength, lenSequence = x.shape
        positions = torch.arange(0, lenSequence).expand(totalLength, lenSequence).to(self.device)
        output = self.dropout((self.wordEmbedding(x) + self.positionalEmbedding(positions)))
        for layer in self.layers:
            output = layer(output, output, output, mask)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, embeddingSize, numberHeads, forwardExp, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embeddingSize)
        self.attention = SelfAttention(embeddingSize, numberHeads=numberHeads)
        self.unitTransformer = TransformerBlock(embeddingSize, numberHeads, dropout, forwardExp)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, sourceMask, targetMask):
        attention = self.attention(x, x, x, targetMask)
        query = self.dropout(self.norm(attention + x))
        output = self.unitTransformer(value, key, query, sourceMask)
        return output


class Decoder(nn.Module):
    def __init__(self,targetVocabSize,embeddingSize,numberLayers,numberHeads,forwardExp,dropout,device,maxLength,):
        super(Decoder, self).__init__()
        self.device = device
        self.wordEmbedding = nn.Embedding(targetVocabSize, embeddingSize)
        self.positionalEmbedding = nn.Embedding(maxLength, embeddingSize)
        self.layers = nn.ModuleList([DecoderBlock(embeddingSize, numberHeads,forwardExp, dropout, device) for _ in range(numberLayers)])
        self.fullyConnected = nn.Linear(embeddingSize, targetVocabSize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encodedOutput, sourceMask, targetMask):
        totalLength, lenSequence = x.shape
        positions = torch.arange(0, lenSequence).expand(totalLength, lenSequence).to(self.device)
        x = self.dropout((self.wordEmbedding(x) + self.positionalEmbedding(positions)))
        for layer in self.layers:
            x = layer(x, encodedOutput, encodedOutput, sourceMask, targetMask)
        output = self.fullyConnected(x)
        return output

class Transformer(nn.Module):
    def __init__(self,sourceVocabSize,targetVocabSize,sourceIndexPad,targetIndexPad,embeddingSize=512,numberLayers=6,forwardExp=4,numberHeads=8,dropout=0,device="cpu",maxLength=100,):
        super(Transformer, self).__init__()
        self.encoder = Encoder(sourceVocabSize,embeddingSize,numberLayers,numberHeads,device,forwardExp,dropout,maxLength,)
        self.decoder = Decoder(targetVocabSize,embeddingSize,numberLayers,numberHeads,forwardExp,dropout,device,maxLength,)
        self.sourceIndexPad = sourceIndexPad
        self.targetIndexPad = targetIndexPad
        self.device = device

    def buildSourceMask(self, source):
        sourceMask = (source != self.sourceIndexPad).unsqueeze(1).unsqueeze(2)
        return sourceMask.to(self.device)

    def buildTargetMask(self, target):
        totalLength, target_len = target.shape
        targetMask = torch.tril(torch.ones((target_len, target_len))).expand(totalLength, 1, target_len, target_len)
        return targetMask.to(self.device)

    def forward(self, source, target):
        sourceMask = self.buildSourceMask(source)
        targetMask = self.buildTargetMask(target)
        enc_source = self.encoder(source, sourceMask)
        output = self.decoder(target, enc_source, sourceMask, targetMask)
        return output
