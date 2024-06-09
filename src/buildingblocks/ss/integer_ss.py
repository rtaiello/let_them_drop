from math import factorial, log2
from os import urandom as rng
from typing import Dict, List

from .shamir_ss import Share


class IShare(Share):
    """Represents an individual share in Shamir's Secret Sharing scheme.

    Attributes:
        bits (int): The number of bits required to represent the share.

    Methods:
        __init__(self, idx, value) -> None: Initializes a new IShare object.
        __add__(self, other): Defines addition operation for shares.
        __mul__(self, other): Defines multiplication operation for shares.
        __sub__(self, other): Defines subtraction operation for shares.
        __mul__scalar__(self, scalar): Defines scalar multiplication operation for shares.
        __div__scalar__(self, scalar): Defines scalar division operation for shares.
        getrealsize(self) -> int: Gets the real size of the share.

    """

    bits: int = 0

    def __init__(self, idx: int, value: int) -> None:
        """Initializes a new IShare object.

        Args:
            idx (int): The index of the share.
            value (int): The value of the share.

        """
        super().__init__(idx, value)

    def __add__(self, other):
        """Defines the addition operation for shares.

        Args:
            other (IShare): Another IShare object.

        Returns:
            IShare: A new IShare object containing the result of the addition.

        Raises:
            AssertionError: If the shares have different indices.

        """
        assert self.idx == other.idx, "Adding shares of different indices"
        return IShare(self.idx, self.value + other.value)

    def __mul__(self, other):
        """Defines the multiplication operation for shares.

        Args:
            other (IShare): Another IShare object.

        Returns:
            IShare: A new IShare object containing the result of the multiplication.

        Raises:
            AssertionError: If the shares have different indices.

        """
        assert self.idx == other.idx, "Multiplying shares of different indices"
        return IShare(self.idx, self.value * other.value)

    def __sub__(self, other):
        """Defines the subtraction operation for shares.

        Args:
            other (IShare): Another IShare object.

        Returns:
            IShare: A new IShare object containing the result of the subtraction.

        Raises:
            AssertionError: If the shares have different indices.

        """
        assert self.idx == other.idx, "Subtracting shares of different indices"
        return IShare(self.idx, self.value - other.value)

    def __mul__scalar__(self, scalar: int):
        """Defines the scalar multiplication operation for shares.

        Args:
            scalar (int): A scalar value.

        Returns:
            IShare: A new IShare object containing the result of scalar multiplication.

        """
        return IShare(self.idx, self.value * scalar)

    def __div__scalar__(self, scalar: int):
        """Defines the scalar division operation for shares.

        Args:
            scalar (int): A scalar value.

        Returns:
            IShare: A new IShare object containing the result of scalar division.

        """
        return IShare(self.idx, self.value // scalar)

    def get_real_size(self) -> int:
        """Gets the real size of the share.

        Returns:
            int: The real size of the share.

        """
        return Share.user_size + self.bits


class ISSS(object):
    """Represents an implementation of the Incremental Secret Sharing Scheme.

    Attributes:
        bitlength (int): The length of the bits for the secret sharing.
        sigma (int): A parameter used for calculating the bits required for the share.

    Methods:
        __init__(self, bitlength, sigma): Initializes a new ISSS object.
        share(self, secret: int, threshold: int, nusers: int) -> List[IShare]: Performs secret sharing.
        lagrange(self, shares: List[IShare], delta: int) -> Dict[int, int]: Calculates Lagrange coefficients.
        reconstruct(self, shares: List[IShare], threshold: int, delta: int, lag_coeffs: Dict[int, int]) -> int: Reconstructs the secret.

    """

    def __init__(self, bitlength: int, sigma: int):
        """Initializes a new ISSS object.

        Args:
            bitlength (int): The length of the bits for the secret sharing.
            sigma (int): A parameter used for calculating the bits required for the share.

        """
        super().__init__()
        self.bitlength = bitlength
        self.sigma = sigma

    def share(self, secret: int, threshold: int, nusers: int) -> List[IShare]:
        """Performs secret sharing.

        Args:
            secret (int): The secret to be shared.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            nusers (int): The total number of shares to generate.

        Returns:
            List[IShare]: A list of IShare objects representing the shares.

        """
        delta: int = factorial(nusers)
        coeffs: List[int] = []
        bits = self.bitlength + log2(delta**2) + self.sigma
        IShare.bits = bits
        nbbytes = int(bits / 8)
        for _ in range(threshold - 1):
            sign = 1
            if int.from_bytes(rng(1), "big") % 2 == 0:
                sign = -1
            coeff = sign * int.from_bytes(rng(nbbytes), "big")
            coeffs.append(coeff)

        coeffs.append(secret * delta)

        def make_share(idx, coeffs):
            share = 0
            for coeff in coeffs:
                share = idx * share + coeff
            return share

        shares = [IShare(i, make_share(i, coeffs)) for i in range(1, nusers + 1)]
        return shares

    def lagrange(self, shares: List[IShare], delta: int) -> Dict[int, int]:
        """Calculates Lagrange coefficients for reconstructing the secret.

        Args:
            shares (List[IShare]): A list of IShare objects representing the shares.
            delta (int): The delta parameter used in the secret sharing.

        Returns:
            Dict[int, int]: A dictionary containing the Lagrange coefficients.

        Raises:
            ValueError: If duplicate shares are provided.

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

    def reconstruct(
        self,
        shares: List[IShare],
        threshold: int,
        delta: int,
        lag_coeffs: Dict[int, int],
    ) -> int:
        """Reconstructs the secret from the given shares using Lagrange interpolation.

        Args:
            shares (List[IShare]): A list of IShare objects representing the shares.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            delta (int): The delta parameter used in the secret sharing.
            lag_coeffs (Dict[int, int], optional): A dictionary containing the Lagrange coefficients.
                Defaults to None.

        Returns:
            int: The reconstructed secret.

        Raises:
            AssertionError: If not enough shares are provided for reconstruction.

        """
        assert len(shares) >= threshold, "Not enough shares, cannot reconstruct!"
        raw_shares = []
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
        return result // delta**2
