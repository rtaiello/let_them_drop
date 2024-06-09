from os import urandom as rng
from typing import List

from ..utils import PField, get_field


class Share(object):
    """Represents an individual share in the Secret Sharing scheme.

    Attributes:
        idx (int): The index of the share.
        value (PField): The value of the share.

    Methods:
        __init__(self, idx: int, value: PField) -> None: Initializes a new Share object.
        __add__(self, other): Defines addition operation for shares.
        __sub__(self, other): Defines subtraction operation for shares.
        __mul__(self, other): Defines multiplication operation for shares.
        __mul__scalar__(self, scalar): Defines scalar multiplication operation for shares.
        __div__scalar__(self, scalar): Defines scalar division operation for shares.
        getrealsize(self) -> int: Gets the real size of the share.

    """

    user_size: int = 16

    def __init__(self, idx: int, value: PField) -> None:
        """Initializes a new Share object.

        Args:
            idx (int): The index of the share.
            value (PField): The value of the share.

        """
        super().__init__()
        self.idx = idx
        self.value = value

    def __add__(self, other):
        """Defines the addition operation for shares.

        Args:
            other (Share): Another Share object.

        Returns:
            Share: A new Share object containing the result of the addition.

        Raises:
            AssertionError: If the shares have different indices.

        """
        assert self.idx == other.idx, "Adding shares of different indices"
        return Share(self.idx, self.value + other.value)

    def __sub__(self, other):
        """Defines the subtraction operation for shares.

        Args:
            other (Share): Another Share object.

        Returns:
            Share: A new Share object containing the result of the subtraction.

        Raises:
            AssertionError: If the shares have different indices.

        """
        assert self.idx == other.idx, "Subtracting shares of different indices"
        return Share(self.idx, self.value - other.value)

    def __mul__(self, other):
        """Defines the multiplication operation for shares.

        Args:
            other (Share): Another Share object.

        Returns:
            Share: A new Share object containing the result of the multiplication.

        Raises:
            AssertionError: If the shares have different indices.

        """
        assert self.idx == other.idx, "Multiplying shares of different indices"
        return Share(self.idx, self.value * other.value)

    def __mul__scalar__(self, scalar):
        """Defines the scalar multiplication operation for shares.

        Args:
            scalar: A scalar value.

        Returns:
            Share: A new Share object containing the result of scalar multiplication.

        """
        scalar_field = PField(scalar, self.value.p, self.value.bits)
        return Share(self.idx, self.value * scalar_field)

    def __div__scalar__(self, scalar):
        """Defines the scalar division operation for shares.

        Args:
            scalar: A scalar value.

        Returns:
            Share: A new Share object containing the result of scalar division.

        """
        scalar_field = PField(scalar, self.value.p, self.value.bits)
        inverse_scalar_field = scalar_field.inverse()
        return Share(self.idx, self.value * inverse_scalar_field)

    def get_real_size(self) -> int:
        """Gets the real size of the share.

        Returns:
            int: The real size of the share.

        """
        return Share.user_size + self.value.bits


class SSS(object):
    """Represents an implementation of the Secret Sharing Scheme.

    Attributes:
        bitlength (int): The length of the bits for the secret sharing.
        Field (PField): The prime field used for the secret sharing.

    Methods:
        __init__(self, bitlength) -> None: Initializes a new SSS object.
        share(self, secret, threshold, nusers): Performs secret sharing.
        reconstruct(self, shares: int, threshold: int, lagcoefs: List[PField]): Reconstructs the secret.
        lagrange(self, shares): Calculates Lagrange coefficients for reconstructing the secret.

    """

    def __init__(self, bitlength: int) -> None:
        """Initializes a new SSS object.

        Args:
            bitlength (int): The length of the bits for the secret sharing.

        """
        super().__init__()
        self.Field = get_field(bitlength)
        self.bitlength = bitlength

    def share(self, secret, threshold, nusers, get_coeffs=False):
        """Performs secret sharing.

        Args:
            secret: The secret to be shared.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            nusers (int): The total number of shares to generate.

        Returns:
            List[Share]: A list of Share objects representing the shares.

        """
        nbbytes = int(self.bitlength / 8)
        coeffs = []
        for _ in range(threshold - 1):
            field_value = self.Field(rng(nbbytes))
            coeffs.append(field_value)
        coeffs.append(self.Field(secret))

        def make_share(user, coeffs):
            idx = self.Field(user)
            share = self.Field(0)
            for coeff in coeffs:
                share = idx * share + coeff
            return share

        shares = []
        for i in range(1, nusers + 1):
            share = make_share(i, coeffs)
            share_obj = Share(i, share)
            shares.append(share_obj)
        if get_coeffs:
            return shares, coeffs
        return shares

    def reconstruct(self, shares: List[Share], threshold: int, lagcoefs: List[PField]):
        """Reconstructs the secret from the given shares using Lagrange interpolation.

        Args:
            shares (List[Share]): A list of Share objects representing the shares.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            lagcoefs (List[PField]): A list containing the Lagrange coefficients.

        Returns:
            int: The reconstructed secret.

        Raises:
            AssertionError: If not enough shares are provided for reconstruction.

        """
        assert len(shares) >= threshold, "Not enough shares, cannot reconstruct!"
        raw_shares = []
        for x in shares:
            idx = self.Field(x.idx)
            value = x.value
            if any(y[0] == idx for y in raw_shares):
                raise ValueError("Duplicate share")
            raw_shares.append((idx, value))
        k = len(shares)
        result = self.Field(0)
        for j in range(k):
            x_j, y_j = raw_shares[j]
            result += y_j * lagcoefs[x_j]
        return result._value

    def lagrange(self, shares: List[Share]):
        """Calculates Lagrange coefficients for reconstructing the secret.

        Args:
            shares (List[Share]): A list of Share objects representing the shares.

        Returns:
            List[PField]: A list containing the Lagrange coefficients.

        Raises:
            ValueError: If duplicate shares are provided.

        """
        k = len(shares)
        indices: List[PField] = []
        for x in shares:
            idx = self.Field(x.idx)
            if any(y == idx for y in indices):
                raise ValueError("Duplicate share")
            indices.append(idx)

        lag_coeffs = {}
        for j in range(k):
            x_j = indices[j]

            numerator = self.Field(1)
            denominator = self.Field(1)

            for m in range(k):
                x_m = indices[m]
                if m != j:
                    numerator *= x_m
                    denominator *= x_m - x_j
            lag_coeffs[x_j] = numerator * denominator.inverse()
        return lag_coeffs
