# tests/test_shamir_ss.py

from math import factorial

import pytest

from src.buildingblocks import ISSS, IShare


class TestIShare:
    def test_add(self):
        share1 = IShare(1, 10)
        share2 = IShare(1, 20)
        result = share1 + share2
        assert result.idx == 1
        assert result.value == 30

    def test_mul(self):
        share1 = IShare(2, 5)
        share2 = IShare(2, 3)
        result = share1 * share2
        assert result.idx == 2
        assert result.value == 15

    def test_getrealsize(self):
        share = IShare(3, 8)
        assert share.get_real_size() == IShare.user_size + IShare.bits


class TestISSS:
    def test_share_and_reconstruct(self):
        secret = 42
        threshold = 3
        nusers = 5
        bitlength = 64
        sigma = 10

        iss = ISSS(bitlength, sigma)

        # Share the secret
        shares = iss.share(secret, threshold, nusers)

        # Make sure the number of shares matches nusers
        assert len(shares) == nusers

        # Reconstruct the secret using a subset of shares
        selected_shares = shares[:threshold]
        delta = factorial(nusers)
        lag_coeffs = iss.lagrange(selected_shares, delta)
        reconstructed_secret = iss.reconstruct(
            selected_shares, threshold, delta, lag_coeffs
        )

        # Make sure the reconstructed secret matches the original secret
        assert reconstructed_secret == secret

    def test_invalid_threshold(self):
        iss = ISSS(64, 10)

        secret = 42
        threshold = 3
        nusers = 5
        bitlength = 64
        sigma = 10

        iss = ISSS(bitlength, sigma)

        # Share the secret
        shares = iss.share(secret, threshold, nusers)

        # Make sure the number of shares matches nusers
        assert len(shares) == nusers

        # Reconstruct the secret using less than the threshold
        selected_shares = shares[: threshold - 1]
        delta = factorial(nusers)
        with pytest.raises(AssertionError):
            lag_coeffs = iss.lagrange(selected_shares, delta)
            _ = iss.reconstruct(selected_shares, threshold, delta, lag_coeffs)


if __name__ == "__main__":
    pytest.main()
