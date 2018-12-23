# -*- encoding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

from io import open

import matplotlib.pyplot as plt
import numpy
import tqdm

from utils import zeros, LongTensor, ByteTensor

plt.switch_backend('agg')
import matplotlib.ticker as ticker

import time
import math
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_IDX = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 30

PAD = "PAD"
SOS = "SOS"
EOS = "EOS"


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: PAD, 1: SOS, 2: EOS}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def normalizeString(s):
    # s = unicodeToAscii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?əıöüğşçƏIÖİĞŞÇÜ]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    with open('data/{l1}-{l2}/{l1}.txt'.format(l1=lang1, l2=lang2), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    pairs = [[normalizeString(text) for text in line.split('\t')] for line in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read {} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            hidden_size,
            batch_first=True,
            num_layers=1,
            bidirectional=False):
        # TODO: maybe use dropout
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            int(hidden_size / (2 if bidirectional else 1)),
            batch_first=batch_first,
            num_layers=num_layers,
            bidirectional=bidirectional)

        self = self.cuda() if torch.cuda.is_available() else self

        # TODO: maybe to be put in reset
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        hidden = self.initHidden(input.shape[0])
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=self.gru.batch_first)
        output, hidden = self.gru(packed, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.gru.batch_first)
        if self.gru.num_layers > 1:
            indices = [self.gru.num_layers - 1]
            if self.gru.bidirectional:
                indices.append(2 * self.gru.num_layers - 1)  # TODO: needs to be verified
        else:
            indices = [0, 1] if self.gru.bidirectional else [0]
        return output, torch.cat([hidden[idx] for idx in indices], dim=1).unsqueeze(0)

    def initHidden(self, batch_size):  # TODO: check if it is needed
        num_directions = 2 if self.gru.bidirectional else 1
        return zeros(
            self.gru.num_layers * num_directions,
            batch_size,
            self.gru.hidden_size)


# TODO: adapt later
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, source_lengths=None):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            outputs of encoder, in shape B x L x H
        :return
            attention energies in shape B x l
        '''
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # B x L x H
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if source_lengths is not None:
            mask = []
            for idx in range(batch_size):
                mask.append([0] * source_lengths[idx].item() + [1] * (max_len - source_lengths[idx].item()))
            mask = ByteTensor(mask)  # B x L
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attention(torch.cat([hidden, encoder_outputs], 2)))  # BxLx2H -> BxLxH
        energy = energy.transpose(2, 1)  # BxHxL
        v = self.v.repeat(encoder_outputs.shape[0], 1).unsqueeze(1)  # Bx1xH
        energy = torch.bmm(v, energy)  # Bx1xL
        return energy.squeeze(1)  # BxL


class AttentionDecoderRNN(nn.Module):
    def __init__(
            self,
            hidden_size,
            embedding_dim,
            num_embeddings,
            dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embedding_dim, hidden_size, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, num_embeddings)
        self = self.cuda() if torch.cuda.is_available() else self

    def forward(self, input, last_hidden, encoder_outputs, source_lengths):
        word_embedded = self.embedding(input).view(1, input.size(0), -1)  # 1, B, H
        word_embedded = self.dropout(word_embedded)

        attn_weights = self.attention(last_hidden[-1], encoder_outputs, source_lengths)
        context = attn_weights.bmm(encoder_outputs)  # Bx1xH
        context = context.transpose(0, 1)  # 1xBxH

        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        output = F.log_softmax(self.out(output))
        return output, hidden


def sentence2sequence(lang, sentence):
    return tuple(lang.word2index[word] for word in sentence.split(' ')) + (EOS_token,)


def tensorsFromPair(pair):
    input = sentence2sequence(input_lang, pair[0])
    target = sentence2sequence(output_lang, pair[1])
    return (input, target)


def train(
        encoder_input,
        input_lengths,
        target_tensor,
        target_lengths,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    encoder_outputs, encoder_hidden = encoder(encoder_input, input_lengths)
    decoder_hidden = encoder_hidden
    decoder_input = LongTensor([SOS_token]).repeat(encoder_input.shape[0], 1)  # one for each instance in batch

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False

    if use_teacher_forcing:  # TODO: adapt teacher forcing
        # Teacher forcing: Feed the target as the next input
        for idx in range(target_lengths.shape[1]):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[idx])
            decoder_input = target_tensor[idx]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        max_target_length = target_lengths.max().item()
        target_lengths_copy = target_lengths.clone()
        for idx in range(max_target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, input_lengths)
            mask = target_lengths_copy > PAD_IDX
            target_lengths_copy -= 1
            masked_output = decoder_output * mask.unsqueeze(1).float()
            topv, topi = masked_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(masked_output[mask], target_tensor[:, idx][mask])
            # or alternative below
            # for instance_idx, target_word in enumerate(target_tensor[:, idx]):
            #     if idx < target_lengths[instance_idx]:
            #         loss += criterion(masked_output[instance_idx].view(1, -1),
            #                           target_word.view(1))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_lengths.sum().item()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def batch_generator(*arrays, batch_size=32, should_shuffle=False):
    input, target = arrays
    if should_shuffle:
        from sklearn.utils import shuffle
        input, target = shuffle(input, target)

    num_instances = len(input)
    batch_count = int(numpy.ceil(num_instances / batch_size))
    progress = tqdm.tqdm(total=num_instances)
    input_length_in_words = numpy.array([len(seq) for seq in input], dtype=numpy.int32)
    target_length_in_words = numpy.array([len(seq) for seq in target], dtype=numpy.int32)

    for idx in range(batch_count):
        startIdx = idx * batch_size
        endIdx = (idx + 1) * batch_size if (idx + 1) * batch_size < num_instances else num_instances

        batch_input_lengths = input_length_in_words[startIdx:endIdx]
        input_maxlength = batch_input_lengths.max()
        input_lengths_argsort = \
            numpy.argsort(batch_input_lengths)[::-1].copy()  # without the copy torch complains about negative strides

        batch_target_lengths = target_length_in_words[startIdx:endIdx]
        target_maxlength = batch_target_lengths.max()

        batch_input = LongTensor([input_seq + (PAD_IDX,) * (input_maxlength - len(input_seq))
                                  for input_seq in input[startIdx:endIdx]])

        batch_target = LongTensor([target_seq + (PAD_IDX,) * (target_maxlength - len(target_seq))
                                   for target_seq in target[startIdx:endIdx]])

        progress.update(len(batch_input_lengths))
        yield batch_input[input_lengths_argsort], LongTensor(batch_input_lengths)[input_lengths_argsort], \
              batch_target[input_lengths_argsort], LongTensor(batch_target_lengths)[input_lengths_argsort]

    progress.close()


def convert_to_sequences(pairs, num_instances):
    return list(zip(*[tensorsFromPair(random.choice(pairs)) for _ in range(num_instances)]))


def trainIters(
        encoder,
        decoder,
        pairs,
        num_instances,
        print_every=1000,
        plot_every=100,
        learning_rate=0.01,
        batch_size=32):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(reduction='sum')
    arrays = convert_to_sequences(pairs, num_instances)

    num_instances_seen = 0
    batch_count = 0
    for batch_input, input_lengths, batch_target, target_lengths \
            in batch_generator(*arrays, batch_size=batch_size, should_shuffle=False):
        loss = train(
            batch_input,
            input_lengths,
            batch_target,
            target_lengths,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion)
        num_instances_seen += batch_input.shape[0]
        batch_count += 1
        print_loss_total += loss
        plot_loss_total += loss

        if num_instances_seen % print_every == 0:
            print_loss_avg = print_loss_total / batch_count
            print_loss_total = 0
            print('{} ({} {}%) {}'.format(timeSince(start, num_instances_seen / num_instances), num_instances_seen,
                                          num_instances_seen / num_instances * 100,
                                          print_loss_avg))
            for _ in range(10):
                sent1, sent2 = random.choice(pairs)
                output_words = evaluate(encoder, decoder, sent1)
                output_sentence = ' '.join(output_words)
                print('input: {}'.format(sent1))
                print('target: {}'.format(sent2))
                print('output: {}'.format(output_sentence))
        #
        # if num_instances_seen % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0
    #
    # showPlot(plot_losses)


def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = {}, loss = {}, time = {}".format(epoch, loss, time))
    if filename and model:
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch{}-{}".format(epoch, loss))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = sentence2sequence(input_lang, sentence)
        input_length = len(input_tensor)

        encoder_input = LongTensor(input_tensor).view(1, -1)
        encoder_outputs, encoder_hidden = encoder(encoder_input, LongTensor([input_length]))

        decoder_hidden = encoder_hidden
        decoder_input = LongTensor([SOS_token]).repeat(encoder_input.shape[0], 1)

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                     LongTensor([input_length]))
            # decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach().view(1, 1)

        return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData('aze', 'eng', True)
    print(random.choice(pairs))
    teacher_forcing_ratio = 0.5
    hidden_size = 256
    input_embedding_dim = 200
    output_embedding_dim = 200

    # encoder = EncoderRNN(input_lang.n_words, input_embedding_dim, hidden_size, num_layers=2, bidirectional=True)
    encoder = EncoderRNN(input_lang.n_words, input_embedding_dim, hidden_size, num_layers=1, bidirectional=False)
    attn_decoder1 = AttentionDecoderRNN(hidden_size, output_embedding_dim, output_lang.n_words, dropout_p=0.1)

    batch_size = 32
    trainIters(encoder, attn_decoder1, pairs, 75000, print_every=64 * batch_size, batch_size=batch_size)
    # evaluateRandomly(encoder, attn_decoder1)
