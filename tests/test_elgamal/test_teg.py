from random import randint

import pytest

from src.buildingblocks import TElGamal


@pytest.mark.parametrize("threshold, nusers", [(1, 5), (3, 5), (5, 5)])
def test_scheme(threshold, nusers):
    teg = TElGamal(threshold, nusers)
    pk, sk_shares = teg.setup()
    message = randint(0, 1000)
    message_point = message * teg.curve_params.p
    enc_point = teg.encrypt(message_point, pk)
    partial_decryptions = [
        teg.share_decrypt(enc_point, sk_share) for sk_share in sk_shares
    ]
    selected_decryptions = partial_decryptions[: teg.threshold]
    lag_coeffs = teg.ss.lagrange(sk_shares[: teg.threshold])
    restored_point = teg.decrypt(selected_decryptions, enc_point, lag_coeffs)
    assert restored_point == message_point
