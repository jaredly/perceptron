#!/usr/bin/env python

import sys
from perceive import fromArff

def main(args):
    fname = 'ex_1.arff'
    rate = .1
    if len(args):
        fname = args[0]
        if len(args) > 1:
            rate = float(args[1])
    m = fromArff(fname, rate)
    a = m.trainUp()
    print '\n'.join(map(str, a))

if __name__ == '__main__':
    main(sys.argv[1:])

# vim: et sw=4 sts=4
