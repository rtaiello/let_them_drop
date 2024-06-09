from math import ceil
from operator import add

import pytest

from src.buildingblocks import TJLS
from src.protocols import ClientFTSA, ServerFTSA


@pytest.mark.parametrize("nclients, dropout_rate", [(32, 0.0), (32, 0.1), (32, 0.3)])
def test_protocol(nclients, dropout_rate):
    dimension = 1000
    inputsize = 16
    keysize = 2048
    dropout = ceil(nclients * dropout_rate)
    threshold = ceil(2 * nclients / 3)
    publicparam, _, _ = TJLS(threshold, nclients).setup(keysize)

    ClientFTSA.set_scenario(
        dimension, inputsize, keysize, threshold, nclients, publicparam
    )
    ServerFTSA.set_scenario(
        dimension, inputsize, keysize, threshold, nclients, publicparam
    )

    clients = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx] = ClientFTSA(idx)

    server = ServerFTSA()

    all_pks = {}
    all_pkc = {}
    for i in range(nclients):
        idx = i + 1
        user, pks, pkc = clients[idx].setup_register()
        all_pks[user] = pks
        all_pkc[user] = pkc

    all_pks, all_pkc = server.setup_register(all_pks, all_pkc)

    all_ek_shares = {}
    for i in range(nclients):
        idx = i + 1
        user, eshares = clients[idx].setup_keysetup(all_pks, all_pkc)
        all_ek_shares[user] = eshares

    all_ek_shares = server.setup_keysetup(all_ek_shares)

    for i in range(nclients):
        idx = i + 1
        clients[idx].setup_keysetup2(all_ek_shares[i + 1])

    all_eb_shares = {}
    all_y = {}
    for i in range(nclients - dropout):
        idx = i + 1
        clients[idx].new_fl_step()
        user, eshares, y = clients[idx].online_encrypt()
        all_eb_shares[user] = eshares
        all_y[user] = y

    all_eb_shares = server.online_encrypt(all_eb_shares, all_y)
    nclients_on = nclients - ceil(nclients * dropout_rate)
    all_y = {idx: y for idx, y in all_y.items() if idx <= nclients_on}
    all_b_shares = {}
    y_zero_shares = {}
    for i in range(nclients - dropout):
        user, bshares, y_zero_share = clients[i + 1].online_construct(
            all_eb_shares[i + 1]
        )

        all_b_shares[user] = bshares
        y_zero_shares[user] = y_zero_share

    sum_protocol = server.online_construct(all_b_shares, y_zero_shares.values())

    sum_clear = clients[1].x

    for i in range(1, nclients - dropout):
        sum_clear = list(map(add, sum_clear, clients[i + 1].x))

    assert sum_protocol == sum_clear
