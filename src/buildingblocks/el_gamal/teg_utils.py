import random
from typing import Iterable, List, Optional

from Crypto.PublicKey import ECC


class PublicParamEG:
    """
    Contains the curve parameters the scheme uses. Since PyCryptodome is used, only curves present there are available:
    https://pycryptodome.readthedocs.io/en/latest/src/public_key/ecc.html
    """

    DEFAULT_CURVE = "P-256"

    def __init__(self, curve_name: str = DEFAULT_CURVE):
        """
        Construct the curve from a given curve name (according to curves present in PyCryptodome).

        :param curve_name:
        """
        assert curve_name in ECC._curves, "Curve {} not present in PyCryptodome".format(
            curve_name
        )

        self._name = curve_name
        self._curve = ECC._curves[curve_name]
        self.p = ECC.EccPoint(x=self._curve.Gx, y=self._curve.Gy, curve=curve_name)

    @property
    def order(self):
        return int(self._curve.order)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._curve == other._curve

    def __str__(self):
        return "Curve {} of order {} with generator point P = {}".format(
            self._name, self.order, self.p
        )


class PublicKeyEG:
    """
    The public key point Q linked to the (implicit) secret key d of the scheme.
    """

    def __init__(self, q: ECC.EccPoint, curve_params: PublicParamEG = PublicParamEG()):
        """
        Construct the public key.

        :param Q: the public key point Q = dP
        :param curve_params: the curve parameters used for constructing the key.
        """
        self.Q = q
        self.curve_params = curve_params

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.curve_params == other.curve_params
            and self.Q == other.Q
        )

    def __str__(self):
        return "Public key point Q = {} (on curve {})".format(
            self.Q, self.curve_params._name
        )


class EncryptedPointEG:
    # TODO include curve_params?
    """
    An encrypted message in the scheme. Because a hybrid approach is used it consists of three parts:

    - C1 = kP as in the ElGamal scheme
    - C2 = kQ + rP as in the ElGamal scheme with rP being the encrypted point for a random value r
    - ciphertext, the symmetrically encrypted message.

    The symmetric key is derived from the ElGamal encrypted point rP.

    Note: The ECIES approach for ECC
    - chooses a random r,
    - computes R=rP and S=rQ,
    - derives a symmetric key k from S,
    - uses R and the symmetric encryption of m as ciphertext.
    But to enable the re-encryption of ciphertexts, here the approach similar to regular ElGamal is used instead.
    """

    def __init__(
        self,
        c1: ECC.EccPoint,
        c2: ECC.EccPoint,
        #  ciphertext: bytes
    ):
        """
        Construct a encrypted message.

        :param v: like in ElGamal scheme
        :param c: like in ElGamal scheme
        :param ciphertext: the symmetrically encrypted message
        """
        self.c1 = c1
        self.c2 = c2

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.c1 == other.c1
            and self.c2 == other.c2
        )

    def __str__(self):
        return "EncryptedMessage (c1, c2) = ({}, {}))".format(self.c1, self.c2)

    def get_real_size(self):
        c1_size_bits = self.c1.size_in_bits()
        c2_size_bits = self.c2.size_in_bits()
        return c1_size_bits + c2_size_bits
