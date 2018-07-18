import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import random
random.seed(1)


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, vocab_num, hidden_dim, label_num):
        super(LSTMClassifier, self).__init__()
        self.word_embedding = nn.Embedding(vocab_num, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()
        self.hidden2label = nn.Linear(hidden_dim, label_num)

    def init_hidden(self):
        return (Variable(torch.randn(1, 1, self.hidden_dim)),
                Variable(torch.randn(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        sent_label = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(sent_label, dim=1)
        return log_probs


class Corpus:
    def __init__(self, filename, maxwordnum):
        # Each line in the file will be: sentence \t words \t polarity \t 1(train)/0(test)
        wordCount = {}  # The number every word appears
        self.wordIDs = {}  # Dictionary that maps words to integers
        self.r_wordIDs = {}  # Inverse of last dictionary
        self.polarity = {"POS": 0, "NEG": 1, "NEU": 2}

        with open(filename, 'r') as f:
            readNum = 0
            for line in f:
                readNum += 1
                if readNum % 5000 == 0:
                    print "Data reading.... Line ", readNum
                msgs = line.strip().split('\t')
                for word in msgs[1].split(' '):
                    try:
                        wordCount[word] += 1
                    except KeyError:
                        wordCount[word] = 1

        sortedword = sorted(wordCount.keys(), key=lambda x: wordCount[x], reverse=True)
        # If the parameter equals to -1, it means there is no word counts limit
        if maxwordnum == -1 or maxwordnum > len(sortedword):
            self.wordNum = len(sortedword)
        else:
            self.wordNum = maxwordnum
        for w in xrange(self.wordNum):
            self.wordIDs[sortedword[w]] = w
            self.r_wordIDs[w] = sortedword[w]

        print "Corpus process over! Word Num: ", self.wordNum


def evaluate(model, test):  # evaluation metrics: accuracy, macro-F1, micro-F1
    TP = [0.0, 0.0, 0.0]    # true positive for POS, NEG, NEU
    TN = [0.0, 0.0, 0.0]    # true negative for POS, NEG, NEU
    FP = [0.0, 0.0, 0.0]    # false positive for POS, NEG, NEU
    FN = [0.0, 0.0, 0.0]    # false negative for POS, NEG, NEU
    total = 0

    for sentence, label in test:
        if torch.cuda.is_available():  # run in GPU
            sentence = sentence.cuda()
            label = label.cuda()
        model.hidden = model.init_hidden()
        predictions = model(sentence)
        pred_label = predictions.data.max(1)[1].numpy()[0]  # the max one is the predicted label
        label = label.data.numpy()[0]
        if pred_label == label:
            if label == 0:
                TP[0] += 1
                TN[1] += 1
                TN[2] += 1
            elif label == 1:
                TN[0] += 1
                TP[1] += 1
                TN[2] += 1
            else:
                TN[0] += 1
                TN[1] += 1
                TP[2] += 1
        else:
            if pred_label == 0:
                FP[0] += 1
                if label == 1:
                    FN[1] += 1
                    TN[2] += 1
                else:
                    FN[2] += 1
                    TN[1] += 1
            elif pred_label == 1:
                FP[1] += 1
                if label == 2:
                    FN[2] += 1
                    TN[0] += 1
                else:
                    FN[0] += 1
                    TN[2] += 1
            else:
                FP[2] += 1
                if label == 0:
                    FN[0] += 1
                    TN[1] += 1
                else:
                    FN[1] += 1
                    TN[0] += 1
        total += 1

    # print TP, TN, FP, FN, total

    acc = (TP[0] + TP[1] + TP[2]) / total

    pre = [0 if TP[0] == 0 else (TP[0] / (TP[0] + FP[0])), 0 if TP[1] == 0 else (TP[1] / (TP[1] + FP[1])),
           0 if TP[2] == 0 else (TP[2] / (TP[2] + FP[2]))]
    rec = [0 if TP[0] == 0 else TP[0] / (TP[0] + FN[0]), 0 if TP[1] == 0 else TP[1] / (TP[1] + FN[1]),
           0 if TP[2] == 0 else TP[2] / (TP[2] + FN[2])]

    # print pre, rec
    pre_ma = sum(pre) / len(pre)
    rec_ma = sum(rec) / len(rec)
    f_ma = 0 if pre_ma == rec_ma == 0 else 2 * pre_ma * rec_ma / (pre_ma + rec_ma)

    pre_mi = 0 if sum(TP) == 0 else sum(TP) / (sum(TP) + sum(FP))
    rec_mi = 0 if sum(TP) == 0 else sum(TP) / (sum(TP) + sum(FN))
    f_mi = 0 if pre_mi == rec_mi == 0 else 2 * pre_mi * rec_mi / (pre_mi + rec_mi)

    res = [acc, f_ma, f_mi]

    return res


def make_sequence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx


def train(corp, training, test):
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 300

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, vocab_num=corp.wordNum,
                           hidden_dim=HIDDEN_DIM, label_num=len(corp.polarity))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if torch.cuda.is_available():              # run in GPU
        model = model.cuda()
        loss_function = loss_function.cuda()

    model.train()
    for epoch in range(EPOCH):
        random.shuffle(training)
        print('Epoch: %d start!' % epoch)
        avg_loss = 0.0
        count = 0
        for sentence, label in training:
            if torch.cuda.is_available():   # run in GPU
                sentence = sentence.cuda()
                label = label.cuda()
            model.zero_grad()
            model.hidden = model.init_hidden()
            predictions = model(sentence)
            # print predictions, label
            loss = loss_function(predictions, label)
            avg_loss += loss.data[0]
            count += 1
            if count % 500 == 0:
                print('Epoch: %d iterations: %d loss :%g' % (epoch, count, loss.data[0]))
            loss.backward()
            optimizer.step()
        avg_loss /= len(training)
        print('Epoch: %d done! \t train avg_loss:%g' % (epoch, avg_loss))

    model.eval()
    res = evaluate(model, test)
    print ('Evaluation result in test set: Accuracy %g, Macro-F1 Score %g, Micro-F1 Score %g' % (res[0], res[1], res[2]))


def run(filename):

    corp = Corpus(filename, -1)
    training = []  # Each element is a set of (sentence, label)
    test = []  # Same as training

    with open(filename, 'r') as f:
        readNum = 0
        for line in f:
            readNum += 1
            if readNum % 5000 == 0:
                print "Data reading.... Line ", readNum
            msgs = line.strip().split('\t')
            if msgs[3] == "1":
                words = msgs[1].split(' ')
                sent = make_sequence(words, corp.wordIDs)
                training.append((sent, make_sequence([msgs[2]], corp.polarity)))
            else:
                words = msgs[1].split(' ')
                sent = make_sequence(words, corp.wordIDs)
                test.append((sent, make_sequence([msgs[2]], corp.polarity)))

    # print test
    train(corp, training, test)

run("sent-test-data.txt")








