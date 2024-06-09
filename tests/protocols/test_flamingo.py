import time
from math import ceil, log2
from operator import add

import pytest

from src.protocols import FlamingoClient, FlamingoServer


@pytest.mark.parametrize(
    "ndecryptors, nclients, dropout_rate", [(10, 32, 0.0), (10, 32, 0.1), (10, 32, 0.3)]
)
def test_protocol(ndecryptors, nclients, dropout_rate):
    dimension = 1000
    inputsize = 16
    keysize = 256
    dropout = ceil(nclients * dropout_rate)
    neighborood_size = ceil((nclients * dropout_rate) / log2(nclients))
    threshold = ceil(2 * ndecryptors / 3)

    FlamingoClient.set_scenario(
        dimension,
        inputsize,
        keysize,
        threshold,
        nclients,
        neighborood_size,
        ndecryptors,
    )
    FlamingoServer.set_scenario(
        dimension,
        inputsize,
        keysize,
        threshold,
        nclients,
        neighborood_size,
        ndecryptors,
    )

    server = FlamingoServer()
    clients = {}
    pub_key, key_shares = server.setup()
    for i in range(nclients):
        idx = i + 1
        if idx <= ndecryptors:
            clients[idx] = FlamingoClient(idx, key_shares[idx - 1], pub_key)
        else:
            clients[idx] = FlamingoClient(idx, 0, pub_key)
    server.new_fl_step()
    all_pks = {}
    all_pkc = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx].new_fl_step()
        user, pks, pkc = clients[i + 1].advertise_keys()
        all_pks[user] = pks
        all_pkc[user] = pkc
    all_pks, all_pkc = server.advertise_keys(all_pks, all_pkc)
    for i in range(nclients):
        idx = i + 1
        clients[idx].report_pairwise_secrets(all_pks, all_pkc)
    all_e_shares = {}
    all_e_messages = {}

    for i in range(nclients):
        idx = i + 1
        user, e_shares, e_messages = clients[idx].report_share_keys()
        all_e_shares[user] = e_shares
        all_e_messages[user] = e_messages
    server.report_share_keys(all_e_shares, all_e_messages)
    all_y = {}

    for i in range(nclients):
        idx = i + 1
        user, y = clients[idx].report_masked_input()
        all_y[user] = y
    nclients_on = nclients - dropout

    all_y = {idx: y for idx, y in all_y.items() if idx <= nclients_on}
    all_e_shares, all_e_messages = server.cross_check(all_y)
    all_b_shares = {}
    all_sk_shares = {}
    for i in range(nclients_on):
        idx = i + 1
        if idx <= ndecryptors:
            user, sk_shares, b_shares = clients[idx].reconstruction(
                all_e_shares[idx], all_e_messages
            )
            all_sk_shares[user] = sk_shares
            all_b_shares[user] = b_shares
    sum_protocol = server.reconstruction(all_sk_shares, all_b_shares)
    sum_clear = [1] * dimension
    for i in range(1, nclients_on):
        sum_clear = list(map(add, sum_clear, [1] * dimension))
    assert sum_protocol == sum_clear
