from Crypto.Hash import SHA256
from gmpy2 import gcd, mpz


class FDH(object):
    def __init__(self, bitsize, N) -> None:
        super().__init__()
        self.bitsize = bitsize
        self.N = N

    def H(self, t):
        counter = 1
        result = b""
        while True:
            while True:
                h = SHA256.new()
                h.update(
                    int(t).to_bytes(self.bitsize // 2, "big")
                    + counter.to_bytes(1, "big")
                )
                result += h.digest()
                counter += 1
                if len(result) < (self.bitsize // 8):
                    break
            r = mpz(int.from_bytes(result[-self.bitsize :], "big"))
            if gcd(r, self.N) == 1:
                break
            else:
                print("HAPPENED")
        return r
