#!/usr/bin/env python

import itertools

import numpy as np
from pandas import DataFrame, Series
from scipy.io.arff import loadarff

class Perceptron:
    def __init__(self, size, rate=.1):
        self.weights = np.zeros(size)
        self.trainingRate = rate

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
    def __init__(self, data, meta, rate=.1, find=None):
        if find is None:
            find = meta.names()[-1]
        length = len(data.columns) - 1
        self.find = find
        self.norm = normalizers(data, meta)
        self.raw = data
        self.data = normalize(self.norm, data)
        self.best = 0

        _, possible = meta[find]

        self.perceptrons = {}
        for truthy, falsy in itertools.combinations(possible, 2):
            self.perceptrons[(truthy, falsy)] = Perceptron(len(data.columns), rate)

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
        return accuracy, weights

    def validate(self, data):
        norm = normalize(self.norm, data)
        for i in norm.index:
            votes = {}
            for (truthy, falsy), p in self.perceptrons.items():
                res, confidence = p.train(norm.loc[i], learn=False)
                vote = truthy if res else falsy
                # at the moment, ignoring confidence
                votes[vote] += 1
            most = sorted(votes.items, (lambda (ka, va), (kb, vb): va - vb))
            print most
            if most[0][0] != norm.loc[

    def trainUp(self):
        history = []
        for i in range(100):
            b = self.best
            for m in range(20):
                # shuffle data
                ix = np.array(self.data.index)
                np.random.shuffle(ix)
                self.data = self.data.reindex(ix)
                # train perceptrons
                history.append(self.train_perceptrons(self.data))
                if self.best == 1:
                    print 'Fully trained'
                    return history
            if self.best == b:
                print 'Done classifying; no progress in past 20 epochs'
                return history[:-20]

def fromArff(fname, rate):
    data, meta = loadarff(fname)
    data = DataFrame(data)
    return Main(data, meta, rate)

# vim: et sw=4 sts=4
