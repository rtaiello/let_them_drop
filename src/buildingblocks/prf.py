import hashlib
from math import ceil

from Crypto.Cipher import ChaCha20
from Crypto.Hash import SHA256
from gmpy2 import mpz

import src.common.crypto as ecchash


class PRF(object):
    _zero = 0
    _nonce = _zero.to_bytes(12, "big")
    security = 256

    def __init__(self, vectorsize, elementsize) -> None:
        super().__init__()
        self.m = vectorsize
        self.bits = elementsize
        self.e = ceil(elementsize / 8)

    def eval_key(self, key, round):
        if isinstance(key, mpz):
            key = int(key)
        if isinstance(key, int):
            if key >= 2**PRF.security:
                mask = 2**PRF.security - 1
                key &= mask
            key = key.to_bytes(PRF.security // 8, "big")
        elif not isinstance(key, bytes):
            raise ValueError("seed should be of type either int or bytes")
        round_number_bytes = round.to_bytes(16, "big")
        c = ChaCha20.new(key=key, nonce=PRF._nonce).encrypt(round_number_bytes)
        c = str(int.from_bytes(c[0:4], "big") & 0xFFFF)

        # map h_ijt to a group element
        dst = ecchash.test_dst("P256_XMD:SHA-256_SSWU_RO_")
        point = ecchash.hash_str_to_curve(
            msg=c,
            count=2,
            modulus=ecchash.n,
            degree=ecchash.m,
            blen=ecchash.L,
            expander=ecchash.XMDExpander(dst, hashlib.sha256, ecchash.k),
        )
        return point

    def eval_vector(self, seed):
        c = ChaCha20.new(key=seed, nonce=PRF._nonce)
        cipher = c.encrypt(b"".rjust(self.e * self.m, b"\x00"))
        return [
            int.from_bytes(cipher[i : i + self.e], "big") % 2**self.bits
            for i in range(0, len(cipher), self.e)
        ]

    def to_hash(self, point):
        key_length = ceil(PRF.security / 8)
        px = int(point.x).to_bytes(key_length, "big")
        py = int(point.y).to_bytes(key_length, "big")

        hash_object = SHA256.new(data=(px + py))
        return hash_object.digest()[0:key_length]
