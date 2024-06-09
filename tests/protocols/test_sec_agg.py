from math import ceil
from operator import add

import pytest

from src.protocols import SecAggClient, SecAggServer


@pytest.mark.parametrize("nclients, dropout_rate", [(32, 0.0), (32, 0.1), (32, 0.3)])
def test_protocol(nclients, dropout_rate):
    dimension = 100
    inputsize = 16
    keysize = 256
    dropout = ceil(nclients * dropout_rate)
    threshold = ceil(2 * nclients / 3)

    SecAggClient.set_scenario(dimension, inputsize, keysize, threshold, nclients)
    SecAggServer.set_scenario(dimension, inputsize, keysize, threshold, nclients)
    clients = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx] = SecAggClient(idx)

    server = SecAggServer()
    all_pks = {}
    all_pkc = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx].new_fl_step()
        user, pks, pkc = clients[i + 1].advertise_keys()
        all_pks[user] = pks
        all_pkc[user] = pkc
    all_pks, all_pkc = server.advertise_keys(all_pks, all_pkc)
    all_e_shares = {}
    for i in range(nclients):
        idx = i + 1
        user, e_shares = clients[idx].share_keys(all_pks, all_pkc)
        all_e_shares[user] = e_shares
    all_e_shares = server.share_keys(all_e_shares)
    all_y = {}
    for i in range(nclients):
        idx = i + 1
        user, y = clients[idx].masked_input_collection(all_e_shares[idx])
        all_y[user] = y
    nclients_on = nclients - dropout
    all_y = {idx: y for idx, y in all_y.items() if idx <= nclients_on}
    clients_on = server.masked_input_collection(all_y)
    all_b_shares = {}
    all_sk_shares = {}
    for i in range(nclients_on):
        user, sk_shares, b_shares = clients[i + 1].unmasking(clients_on)
        all_b_shares[user] = b_shares
        all_sk_shares[user] = sk_shares
    sum_protocol = server.unmasking(all_sk_shares, all_b_shares)
    sum_clear = [1] * dimension
    for i in range(1, nclients_on):
        sum_clear = list(map(add, sum_clear, [1] * dimension))
    assert sum_protocol == sum_clear
