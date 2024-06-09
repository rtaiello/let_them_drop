# tests/test_sss.py

from math import factorial
from os import urandom as rng
from typing import List

import pytest

from src.buildingblocks import SSS, Share
from src.buildingblocks.utils import get_field

field = get_field(64)


class TestShare:
    def test_add(self):
        share1 = Share(1, field(10))
        share2 = Share(1, field(20))
        result = share1 + share2
        assert result.idx == 1
        assert result.value._value == 30

    def test_mul(self):
        share1 = Share(2, field(5))
        share2 = Share(2, field(3))
        result = share1 * share2
        assert result.idx == 2
        assert result.value._value == 15

    def test_getrealsize(self):
        share = Share(3, field(8))
        assert share.get_real_size() == Share.user_size + share.value.bits


class TestSSS:
    def test_share_and_reconstruct(self):
        secret = 42
        threshold = 3
        nusers = 5
        bitlength = 64

        sss = SSS(bitlength)

        # Share the secret
        shares = sss.share(secret, threshold, nusers)

        # Make sure the number of shares matches nusers
        assert len(shares) == nusers

        # Reconstruct the secret using a subset of shares
        selected_shares = shares[:threshold]
        lag_coeffs = sss.lagrange(selected_shares)
        reconstructed_secret = sss.reconstruct(selected_shares, threshold, lag_coeffs)

        # Make sure the reconstructed secret matches the original secret
        assert reconstructed_secret == secret

    def test_invalid_threshold(self):
        bitlength = 64

        sss = SSS(bitlength)

        secret = 42
        threshold = 3
        nusers = 5

        # Share the secret
        shares = sss.share(secret, threshold, nusers)

        # Make sure the number of shares matches nusers
        assert len(shares) == nusers

        # Reconstruct the secret using less than the threshold
        selected_shares = shares[: threshold - 1]
        with pytest.raises(AssertionError):
            lag_coeffs = sss.lagrange(selected_shares)
            _ = sss.reconstruct(selected_shares, threshold, lag_coeffs)


if __name__ == "__main__":
    pytest.main()
