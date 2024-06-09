from os import urandom as rng
from typing import Dict, List, Union

from gmpy2 import mpz, powmod

from ..joye_libert.jl_utils import EncryptedNumberJL
from .shamir_ss import Share

user_size: int = 16


class ShoupShare(Share):
    bits: int = 0

    def __init__(self, idx: int, value: int) -> None:
        """
        Initialize a ShoupShare object.

        Args:
            idx (int): The index of the share.
            value (int): The value of the share.
        """
        super().__init__(idx, value)

    def __add__(self, other):
        """
        Add two ShoupShare objects.

        Args:
            other (ShoupShare): The other share to add.

        Returns:
            ShoupShare: The result of the addition.
        """
        assert self.idx == other.idx, "Adding shares of different indices"
        return ShoupShare(self.idx, self.value + other.value)

    def get_real_size(self) -> int:
        """
        Get the size of the share in bits.

        Returns:
            int: The size of the share in bits.
        """
        return Share.user_size + ShoupShare.bits


class ShoupSSS(object):
    def __init__(self, n: int):
        """
        Initialize a ShoupSSS object.

        Args:
            n (int): The parameter 'n' used in the secret sharing scheme.
        """
        super().__init__()
        self.n = n
        self.n_squared = n**2

    def share(
        self, secret: int, threshold: int, nusers: int, phi_n: int
    ) -> List[ShoupShare]:
        """
        Shares a secret using Shoup Secret Sharing.

        Args:
            secret (int): The secret to share.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            nusers (int): The total number of shares to generate.
            phi_n (int): The totient of n.

        Returns:
            List[ShoupShare]: A list of ShoupShare objects representing the shares.
        """
        bits: int = self.n.bit_length()
        ShoupShare.bits = bits
        nbbytes: int = int(bits / 8)
        coeffs: List[int] = [
            int.from_bytes(rng(nbbytes), "big") for _ in range(threshold - 1)
        ]
        coeffs.append(secret)

        def make_share(idx, coeffs):
            share = 0
            for coeff in coeffs:
                share = idx * share + coeff
            return share % (self.n * phi_n)

        shares: List[ShoupShare] = [
            ShoupShare(i, make_share(i, coeffs)) for i in range(1, nusers + 1)
        ]
        return shares

    def reconstruct(
        self,
        shares: List[ShoupShare],
        threshold: int,
        delta: int,
        lag_coeffs: Dict[int, int] = None,
    ) -> int:
        """
        Reconstructs the secret from the given shares using Lagrange interpolation.

        Args:
            shares (List[ShoupShare]): A list of ShoupShare objects representing the shares.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            delta (int): The delta parameter used in the secret sharing.
            lag_coeffs (Dict[int, int], optional): A dictionary containing the Lagrange coefficients.
                Defaults to None.

        Returns:
            int: The reconstructed secret.
        """
        assert len(shares) >= threshold, "Not enough shares, cannot reconstruct!"
        raw_shares: List[tuple] = []
        for x in shares:
            idx = x.idx
            value = x.value
            if any(y[0] == idx for y in raw_shares):
                raise ValueError("Duplicate share")
            raw_shares.append((idx, value))
        k = len(shares)
        result = 0
        for j in range(k):
            x_j, y_j = raw_shares[j]
            r = y_j * lag_coeffs[x_j]
            result += r
        return result // delta

    def lagrange(self, shares: List[ShoupShare], delta: int) -> Dict[int, int]:
        """
        Calculates Lagrange coefficients for reconstructing the secret.

        Args:
            shares (List[ShoupShare]): A list of ShoupShare objects representing the shares.
            delta (int): The delta parameter used in the secret sharing.

        Returns:
            Dict[int, int]: A dictionary containing the Lagrange coefficients.
        """
        k: int = len(shares)
        indices: List[int] = []
        for x in shares:
            idx = x.idx
            if any(y == idx for y in indices):
                raise ValueError("Duplicate share")
            indices.append(idx)

        lag_coeffs = {}
        for j in range(k):
            x_j = indices[j]

            numerator = 1
            denominator = 1

            for m in range(k):
                x_m = indices[m]
                if m != j:
                    numerator *= x_m
                    denominator *= x_m - x_j
            lag_coeffs[x_j] = (delta * numerator) // denominator
        return lag_coeffs
