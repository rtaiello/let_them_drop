import random
from random import randint

import gmpy2


def add_vectors(xs, ys, mod=None):
    zs = []
    for a, b in zip(xs, ys):
        z = a + b
        if mod:
            z = z % mod
        zs.append(z)
    return zs


def multiply_vector_by_scalar(xs, k, mod=None):
    zs = []
    for a in xs:
        z = a * k
        if mod:
            z = z % mod
        zs.append(z)
    return zs


def sub_vectors(xs, ys, mod=None):
    zs = []
    for a, b in zip(xs, ys):
        z = a - b
        if mod:
            z = z % mod
        zs.append(z)
    return zs


def get_field(bitlength):
    if bitlength <= 64:
        P64Field.bits = bitlength
        field = P64Field
    elif bitlength <= 128:
        P128Field.bits = bitlength
        field = P128Field
    elif bitlength <= 256:
        P256Field.bits = bitlength
        field = P256Field
    elif bitlength <= 512:
        P512Field.bits = bitlength
        field = P512Field
    elif bitlength <= 1024:
        P1024Field.bits = bitlength
        field = P1024Field
    elif bitlength <= 2048:
        P2048Field.bits = bitlength
        field = P2048Field
    else:
        raise ValueError("No sufficient field for this secret size")
    return field


def get_two_safe_primes(bitlength):
    if bitlength == 2048:
        p = 7612036456999683599077717132671455415031908407114945567931900462632365597206721501244304073075221192967828168004574853215746650933832090986220158139468003
        q = 9952622069270053370380344155062236088413830836754885113824514979258221546321394097768536763954080795280385430886254765458716253805918045968364867081064203
    elif bitlength == 4096:
        p = 156171196023471764965930038068236212130742833720153106835167216676997394968416508970275487227265027714772581140817321565789192429799091434620901203089139350978272599315268832422099774313791790176366729425577463186294817143877359666951633393929590150360560691713479213994634278538260551525751878787457936150167
        q = 175791022837015142645179339814548331030891265255834441521864048307932186591750660041545905708601370557207685171431174852277287144826585032161845199188668231553079242457012929889570445049558227948536468242116083725345169686579402487015565178485960854816230484108917304927570171054986177479328522459951938194267
    else:
        raise ValueError("bitlength should be 2048 or 4096")
    return p, q


def create_mask(bitlength, dimension):
    field = get_field(bitlength)
    mask = [
        field(random.SystemRandom().getrandbits(bitlength)) for _ in range(dimension)
    ]
    # mask = [field(d) for d in range(dimension)]
    return mask


def create_random_vector(bitlength, valuesize, dimension):
    field = get_field(bitlength)
    vector = [field(1) for _ in range(dimension)]
    # vector =[field(d) for d in range(dimension)] #
    return vector


def getprimeover(bits):
    randfunc = random.SystemRandom()
    # fix the seed to have reproducible results
    r = gmpy2.mpz(randfunc.getrandbits(bits))
    r = gmpy2.bit_set(r, bits - 1)
    return gmpy2.next_prime(r)


def invert(a, b):
    s = gmpy2.invert(a, b)
    if s == 0:
        raise ZeroDivisionError("invert() no inverse exists")
    return s


def powmod(a, b, c):
    if a == 1:
        return 1
    return gmpy2.powmod(a, b, c)


class PField(object):
    def __init__(self, encoded_value, p, bits):
        self.p = p
        self.bits = bits
        if isinstance(encoded_value, gmpy2.mpz):
            self._value = encoded_value
        elif isinstance(encoded_value, int):
            self._value = gmpy2.mpz(encoded_value)
        elif isinstance(encoded_value, bytes):
            self._value = gmpy2.mpz(int.from_bytes(encoded_value, "big"))
        else:
            raise ValueError(
                "The encoded value is of type {} but it must be an integer or a byte string".format(
                    type(encoded_value)
                )
            )

    def __eq__(self, other):
        return self._value == other._value

    def __int__(self):
        return self._value

    def __hash__(self):
        return self._value.__hash__()

    def encode(self):
        return self._value.to_bytes(256, "big")

    def __mul__(self, factor):
        return PField((self._value * factor._value) % self.p, self.p, self.bits)

    def __add__(self, term):
        return PField((self._value + term._value) % self.p, self.p, self.bits)

    def __sub__(self, term):
        return PField((self._value - term._value) % self.p, self.p, self.bits)

    def __div__(self, term):
        # use the inverse
        return PField(
            (self._value * invert(term._value, self.p)) % self.p, self.p, self.bits
        )

    def inverse(self):
        if self._value == 0:
            raise ValueError("Inversion of zero")

        return PField(invert(self._value, self.p), self.p, self.bits)

    def __pow__(self, exponent):
        return PField(powmod(self._value, exponent._value, self.p), self.p, self.bits)

    def __mod__(self, mod):
        return PField(self._value % mod._value, self.p, self.bits)

    def get_real_size(self):
        return (self._value.bit_length() + 7) // 8

    def __repr__(self):
        return self._value.__repr__()


class P2048Field(PField):
    bits = 2048

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**2203 - 1, P2048Field.bits)


class P1024Field(PField):
    bits = 1024

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**1279 - 1, P1024Field.bits)


class P512Field(PField):
    bits = 512

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**521 - 1, P512Field.bits)


class P256Field(PField):
    # 2**n - k
    bits = 256

    def __init__(self, encoded_value):
        super().__init__(
            encoded_value,
            115792089210356248762697446949407573529996955224135760342422259061068512044369,
            P256Field.bits,
        )


class P128Field(PField):
    # 2**n - k
    bits = 128

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**129 - 1365, P128Field.bits)


class P64Field(PField):
    # 2**n - k
    bits = 64

    def __init__(self, encoded_value):
        super().__init__(encoded_value, 2**65 - 493, P64Field.bits)
