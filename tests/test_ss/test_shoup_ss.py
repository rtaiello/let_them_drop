# tests/test_shamir_shoup_ss.py

from math import factorial

import pytest
from gmpy2 import is_prime, powmod

from src.buildingblocks import ShoupShare, ShoupSSS


class TestShoupShare:
    def test_add(self):
        share1 = ShoupShare(1, 10)
        share2 = ShoupShare(1, 20)
        result = share1 + share2
        assert result.idx == 1
        assert result.value == 30

    def test_getrealsize(self):
        share = ShoupShare(3, 8)
        assert share.get_real_size() == ShoupShare.user_size + ShoupShare.bits


class TestShoupSS:
    secret = 5
    threshold = 3
    nusers = 5
    delta = factorial(nusers)
    bitlength = 64
    p = 65063
    pp = (p - 1) // 2
    q = 127727
    qq = (q - 1) // 2
    p = 2 * pp + 1
    q = 2 * qq + 1
    assert is_prime(p) and is_prime(q) and is_prime(pp) and is_prime(qq)
    n = p * q
    phi_n = pp * qq
    shoup_ss = ShoupSSS(n)

    def test_share_and_reconstruct(self):
        # Share the secret
        shares = TestShoupSS.shoup_ss.share(
            TestShoupSS.secret,
            TestShoupSS.threshold,
            TestShoupSS.nusers,
            TestShoupSS.phi_n,
        )

        # Make sure the number of shares matches nusers
        assert len(shares) == TestShoupSS.nusers

        # Reconstruct the secret using a subset of shares
        selected_shares = shares[: TestShoupSS.threshold]
        lag_coeffs = TestShoupSS.shoup_ss.lagrange(selected_shares, TestShoupSS.delta)
        reconstructed_secret = TestShoupSS.shoup_ss.reconstruct(
            selected_shares, TestShoupSS.threshold, TestShoupSS.delta, lag_coeffs
        )

        # Make sure the reconstructed secret matches the original secret
        assert reconstructed_secret == TestShoupSS.secret

    def test_invalid_threshold(self):
        # Share the secret
        shares = TestShoupSS.shoup_ss.share(
            TestShoupSS.secret,
            TestShoupSS.threshold,
            TestShoupSS.nusers,
            TestShoupSS.phi_n,
        )

        # Make sure the number of shares matches nusers
        assert len(shares) == TestShoupSS.nusers

        # Reconstruct the secret using less than the threshold
        selected_shares = shares[: TestShoupSS.threshold - 1]
        with pytest.raises(AssertionError):
            lag_coeffs = TestShoupSS.shoup_ss.lagrange(
                selected_shares, TestShoupSS.delta
            )
            _ = TestShoupSS.shoup_ss.reconstruct(
                selected_shares, TestShoupSS.threshold, TestShoupSS.delta, lag_coeffs
            )


if __name__ == "__main__":
    pytest.main()
