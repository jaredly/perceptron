#!/usr/bin/env python

import itertools

import numpy as np
from pandas import DataFrame, Series
from scipy.io.arff import loadarff

class Perceptron:
    def __init__(self, size, rate=.1):
        self.weights = np.zeros(size)
        self.trainingRate = rate

    def classify(self, line):
        line = np.concatenate((line, [1]))
        net = self.weights.dot(line)
        output = 1 if net > 0 else 0
        return output, net

    def train(self, line, learn=True):
        # history?
        values = line.copy()
        values[-1] = 1
        net = self.weights.dot(values)
        output = 1 if net > 0 else 0
        target = line[-1]
        if output == target:
            return 1, net
        if not learn:
            return 0, net
        self.learn(values, target, output)
        return 0, net

    def learn(self, values, target, output):
        self.weights += self.trainingRate * (target - output) * values

def normalizers(data, meta):
    ranges = {}
    # print data
    # print meta
    for name in meta.names()[:-1]:
        typ, rng = meta[name]
        if typ == 'numeric':
            mn = data[name].min()
            mx = data[name].max()
            ranges[name] = typ, (mn, mx - mn)
        else:
            ranges[name] = typ, data[name].unique()
    # print ranges
    return ranges

def denormalize(norms, data):
    data = data.copy()
    for name, (typ, more) in norms.items():
        if typ == 'numeric':
            mn, rng = more
            data[name] *= rng
            data[name] += mn
        else:
            total = len(more) - 1.0
            final = np.zeros(len(data[name]), str)
            for i, v in enumerate(more):
                num = i / total
                final[data[name] == num] = v
            data[name] = final
    return data

def normalize(norms, data):
    data = data.copy()
    for name, (typ, more) in norms.items():
        if typ == 'numeric':
            mn, rng = more
            data[name] -= mn
            data[name] /= rng
        else:
            total = len(more) - 1.0
            final = np.zeros(len(data[name]), float)
            for i, v in enumerate(more):
                num = i / total
                final[data[name] == v] = num
            data[name] = final
    return data

def subData(data, truthy, falsy, find):
    ci = list(data.columns).index(find)
    cols = data.columns[:ci] + data.columns[ci + 1:]
    # print truthy, falsy, find
    # print data
    tdata = data[(data[find] == truthy) + (data[find] == falsy)]
    targets = np.zeros(len(tdata[find]), int)
    # print tdata
    # print len(tdata[find]), 'len'
    # print targets
    targets[tdata[find] == truthy] = 1
    targets[tdata[find] == falsy] = 0
    tdata = tdata[cols]
    tdata[find] = targets
    return tdata

class Main:
    '''Perceptron tester

    This will train a perceptron (or mutliple, if there are more than two
    possible outputs) given a dataset and meta information, assumed to come
    from an .arff file.

    When training, if no progress is made in 20 epochs it will quit.

    '''

    def __init__(self, meta, rate=.1, find=None):
        if find is None:
            find = meta.names()[-1]
        length = len(meta.names())
        self.find = find
        self.best = 0
        self.best_weights = None
        self.norm = None
        self.meta = meta

        _, possible = meta[find]

        self.perceptrons = {}
        for truthy, falsy in itertools.combinations(possible, 2):
            self.perceptrons[(truthy, falsy)] = Perceptron(length, rate)

    def train_perceptrons(self, data):
        accuracy = []
        weights = []
        for (truthy, falsy), p in self.perceptrons.items():
            tdata = subData(data, truthy, falsy, self.find)
            dlen = tdata.shape[0]
            # print tdata
            # print tdata.index
            results = []
            for i in tdata.index:
                results.append(p.train(tdata.loc[i])[0])
            # print results
            accuracy.append(sum(results)/float(len(results)))
            weights.append(p.weights.copy())
        total = sum(accuracy)/float(len(accuracy))
        if total > self.best:
            self.best = total
            self.best_weights = weights
        return accuracy, weights

    def validate(self, norm):
        '''norm is already normalized'''
        total = len(norm.index)
        wrong = 0.0
        for i in norm.index:
            votes = {}
            for (truthy, falsy), p in self.perceptrons.items():
                res, confidence = p.classify(norm.loc[i][:-1])
                # print res, confidence, truthy, falsy, p.weights, norm.loc[i]
                vote = truthy if res else falsy
                if vote not in votes:
                    votes[vote] = [0, 0]
                votes[vote][0] += 1
                votes[vote][1] += abs(confidence)
            most = sorted(votes.items(), (lambda (ka, va), (kb, vb): vb[0] - va[0]))
            # print most
            if most[0][0] != norm.loc[i][self.find]:
                # print 'Wrong!'
                wrong += 1
        return wrong/total, wrong

    def trainUp(self, data):
        history = []
        bests = [0]
        for i in range(100):
            # shuffle data
            ix = np.array(data.index)
            np.random.shuffle(ix)
            data = data.reindex(ix)
            # train perceptrons
            history.append(self.train_perceptrons(data))
            if self.best == 1:
                print 'Fully trained'
                return history
            bests.append(self.best)
            if len(bests) > 20:
                if bests.pop(0) == self.best:
                    print 'Done classifying; no progress in past 20 epochs'
                    # revert to the best weights
                    for w, (k, p) in zip(self.best_weights, self.perceptrons.items()):
                        p.weights = w
                    return history

    def train(self, raw, split=None):
        self.norm = normalizers(raw, self.meta)
        data = normalize(self.norm, raw)

        if not split:
            return self.trainUp(data)

        ix = np.array(data.index)
        np.random.shuffle(ix)
        data = data.reindex(ix)

        ln = len(data.index)
        stop = int(ln * split)
        print 'Using', stop, 'for training, and', ln - stop, 'for testing'
        history = self.trainUp(data.loc[data.index[:stop]])
        result = self.validate(data.loc[data.index[stop:]])
        return history, result

def fromArff(fname, rate):
    data, meta = loadarff(fname)
    data = DataFrame(data)
    return data, Main(meta, rate)

# vim: et sw=4 sts=4
