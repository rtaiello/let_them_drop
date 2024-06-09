from os import urandom as rng
from random import randint
from typing import List, Optional, Tuple

from Crypto.PublicKey import ECC
from gmpy2 import powmod

from ..ss.shamir_ss import SSS, Share
from .teg_utils import EncryptedPointEG, PublicKeyEG, PublicParamEG


class TElGamal:
    def __init__(self, threshold: int, nusers: int):
        """
        Initializes a TElGamal instance.

        Args:
            threshold (int): The threshold for secret sharing.
            nusers (int): The number of users.
        """
        self.threshold = threshold
        self.nusers = nusers
        self.curve_params = PublicParamEG()
        # we set the prime order of the curve as the order of the group for the secret sharing of 256 bits
        self.ss = SSS(256)

    def setup(self) -> Tuple[PublicKeyEG, List[Share]]:
        """
        Creates a public key and n shares by choosing a random secret key and using it for computations.

        Returns:
            Tuple[PublicKeyEG, List[Share]]: The public key and n key shares.
        """
        d = randint(1, self.curve_params.order)
        q = d * self.curve_params.p
        pk = PublicKeyEG(q, self.curve_params)

        sk_shares = self.ss.share(d, self.threshold, self.nusers)
        return pk, sk_shares

    def encrypt(self, point: ECC.EccPoint, public_key: PublicKeyEG) -> EncryptedPointEG:
        """
        Encrypts a point using the ElGamal scheme.

        Args:
            point (ECC.EccPoint): The point to be encrypted.
            public_key (PublicKeyEG): The public key for encryption.

        Returns:
            EncryptedPointEG: The encrypted point.
        """
        enc_randomness = randint(1, self.curve_params.order)
        base_point = self.curve_params.p
        c1 = enc_randomness * base_point
        c2 = point + (enc_randomness * public_key.Q)
        return EncryptedPointEG(c1, c2)

    def share_decrypt(
        self, enc_point: EncryptedPointEG, key_share: Share
    ) -> ECC.EccPoint:
        """
        Compute the partial decryption of an encrypted message using a key share.

        Args:
            enc_point (EncryptedPointEG): The encrypted point.
            key_share (Share): The key share.

        Returns:
            ECC.EccPoint: The partial decryption.
        """
        y_c1 = enc_point.c1 * key_share.value._value
        return y_c1

    def decrypt(
        self,
        partial_decryptions: List[ECC.EccPoint],
        enc_point: EncryptedPointEG,
        lag_coeffs,
    ) -> ECC.EccPoint:
        """
        Decrypts an encrypted point using partial decryptions and Lagrange coefficients.

        Args:
            partial_decryptions (List[ECC.EccPoint]): List of partial decryptions.
            enc_point (EncryptedPointEG): The encrypted point.
            lag_coeffs: Lagrange coefficients.

        Returns:
            ECC.EccPoint: The decrypted point.
        """
        assert (
            len(partial_decryptions) >= self.threshold
        ), "Not enough partial decryptions given"

        def _ecc_sum(points: List[ECC.EccPoint]) -> Optional[ECC.EccPoint]:
            """Compute the sum of a list of EccPoints."""
            result = points[0].copy()
            for point in points[1:]:
                result += point
            return result

        summands = []
        for point, lag_coeff in zip(partial_decryptions, lag_coeffs.values()):
            summands.append(point * lag_coeff._value)
        restored_kd_p = _ecc_sum(summands)
        restored_point = enc_point.c2 + (-restored_kd_p)
        return restored_point
