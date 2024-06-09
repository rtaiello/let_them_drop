"""Code Implemented by:
https://github.com/MohamadMansouri/fault-tolerant-secure-agg/tree/main
"""
from gmpy2 import invert, mpz, powmod


class PublicParamJL:
    def __init__(self, n, bits, H):
        super().__init__()
        self.n = n
        self.n_squared = n * n
        self.bits = bits
        self.H = H

    def __eq__(self, other):
        return self.n == other.n

    def __repr__(self):
        hashcode = hex(hash(self.H))
        nstr = self.n.digits()
        return "<PublicParam (N={}...{}, H(x)={})>".format(
            nstr[:5], nstr[-5:], hashcode[:10]
        )


class UserKeyJL(object):
    def __init__(self, param, key):
        super().__init__()
        self.pp = param
        self.s = key

    def __repr__(self):
        hashcode = hex(hash(self))
        return "<UserKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.pp == other.pp and self.s == other.s

    def __hash__(self):
        return hash(self.s)

    def encrypt(self, plaintext, tau):
        if isinstance(plaintext, list):
            counter = 0
            cipher = []
            for pt in plaintext:
                cipher.append(self._encrypt(pt, (counter << self.pp.bits // 2) | tau))
                counter += 1
        else:
            cipher = self._encrypt(plaintext, tau)
        return cipher

    def _encrypt(self, plaintext, tau):
        nude_ciphertext = (self.pp.n * plaintext + 1) % self.pp.n_squared
        r = powmod(self.pp.H(tau), self.s, self.pp.n_squared)
        ciphertext = (nude_ciphertext * r) % self.pp.n_squared
        return EncryptedNumberJL(self.pp, ciphertext)


class ServerKeyJL(object):
    def __init__(self, param, key):
        super().__init__()
        self.pp = param
        self.s = key

    def __repr__(self):
        hashcode = hex(hash(self))
        return "<ServerKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.pp == other.pp and self.s == other.s

    def __hash__(self):
        return hash(self.s)

    def decrypt(self, cipher, tau, delta=1, ttp=False):
        if isinstance(cipher, list):
            counter = 0
            pt = []
            for c in cipher:
                pt.append(
                    self._decrypt(c, (counter << self.pp.bits // 2) | tau, delta, ttp)
                )
                counter += 1
        else:
            pt = self._decrypt(cipher, tau, delta, ttp)
        return pt

    def _decrypt(self, cipher, tau, delta=1, ttp=False):
        if not isinstance(cipher, EncryptedNumberJL):
            raise TypeError("Expected encrypted number type but got: %s" % type(cipher))
        if self.pp != cipher.pp:
            raise ValueError(
                "encrypted_number was encrypted against a " "different key!"
            )
        return self._raw_decrypt(cipher.ciphertext, tau, delta, ttp)

    def _raw_decrypt(self, ciphertext, tau, delta=1, ttp=False):
        if not ttp and delta != 1:
            delta = delta**2
        if not isinstance(ciphertext, mpz):
            raise TypeError(
                "Expected mpz type ciphertext but got: %s" % type(ciphertext)
            )
        V = (
            ciphertext * powmod(self.pp.H(tau), delta * self.s, self.pp.n_squared)
        ) % self.pp.n_squared
        X = ((V - 1) // self.pp.n) % self.pp.n
        X = (X * invert(delta, self.pp.n_squared)) % self.pp.n
        return int(X)


class EncryptedNumberJL:
    def __init__(self, param, ciphertext):
        super().__init__()
        self.pp = param
        self.ciphertext = ciphertext

    def __add__(self, other):
        if isinstance(other, EncryptedNumberJL):
            return self._add_encrypted(other)
        if isinstance(other, mpz):
            e = EncryptedNumberJL(self.pp, other)
            return self._add_encrypted(e)

    def __iadd__(self, other):
        if isinstance(other, EncryptedNumberJL):
            return self._add_encrypted(other)
        if isinstance(other, mpz):
            e = EncryptedNumberJL(self.pp, other)
            return self._add_encrypted(e)

    def __repr__(self):
        estr = self.ciphertext.digits()
        return "<EncryptedNumber {}...{}>".format(estr[:5], estr[-5:])

    def _add_encrypted(self, other):
        if self.pp != other.pp:
            raise ValueError(
                "Attempted to add numbers encrypted against " "different prameters!"
            )

        return EncryptedNumberJL(
            self.pp, self.ciphertext * other.ciphertext % self.pp.n_squared
        )

    def get_real_size(self):
        return self.pp.bits * 2
