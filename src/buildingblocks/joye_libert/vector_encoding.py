"""Code Implemented by:
https://github.com/MohamadMansouri/fault-tolerant-secure-agg/tree/main
"""
from math import ceil, floor, log2

import gmpy2


class VES(object):
    def __init__(self, ptsize, addops, valuesize, vectorsize) -> None:
        super().__init__()
        self.ptsize = ptsize
        self.addops = addops
        self.valuesize = valuesize
        self.vectorsize = vectorsize
        self.elementsize = valuesize + ceil(log2(addops))
        self.compratio = floor(ptsize / self.elementsize)
        self.numbatches = ceil(self.vectorsize / self.compratio)
        self.encoded_size = None

    def encode(self, V):
        bs = self.compratio
        e = []
        E = []
        for v in V:
            e.append(v)
            bs -= 1
            if bs == 0:
                E.append(self._batch(e))
                e = []
                bs = self.compratio
        if e:
            E.append(self._batch(e))
        self.encoded_size = len(E)
        return E

    def decode(self, E):
        V = []
        for e in E:
            for v in self._debatch(e):
                V.append(v)
        return V

    def _batch(self, V):
        i = 0
        a = 0
        for v in V:
            a |= v << self.elementsize * i
            i += 1
        return gmpy2.mpz(a)

    def _debatch(self, b):
        V = []
        bit = 0b1
        mask = 0b1
        for _ in range(self.elementsize - 1):
            mask <<= 1
            mask |= bit

        while b != 0:
            v = mask & b
            V.append(int(v))
            b >>= self.elementsize
        return V
