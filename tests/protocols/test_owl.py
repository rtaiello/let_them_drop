from math import ceil
from typing import Dict

import pytest

from src.buildingblocks import JLS
from src.protocols import OwlClient, OwlServer


@pytest.mark.parametrize("nclients, dropout_rate", [(32, 0.0), (32, 0.1), (32, 0.3)])
def test_protocol(nclients, dropout_rate):
    dimension = 1000
    threshold = 7
    inputsize = 16
    keysize = 2048
    nclients = 10
    pp, _, _ = JLS(nclients).setup(keysize)
    dropout = ceil(nclients * dropout_rate)
    OwlClient.set_scenario(dimension, inputsize, keysize, threshold, nclients, pp)
    OwlServer.set_scenario(dimension, inputsize, keysize, threshold, nclients, pp)

    clients: Dict[int, OwlClient] = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx] = OwlClient(idx)

    server = OwlServer()

    allpkc = {}
    for i in range(nclients):
        user, pkc = clients[i + 1].setup_register()
        allpkc[user] = pkc
    allpkc = server.setup_register(allpkc)

    for i in range(nclients):
        clients[i + 1].setup_keysetup(allpkc)
    eshares = {}
    for i in range(nclients):
        user, eshare = clients[i + 1].online_key_generation()
        eshares[user] = eshare

    eshares = server.online_key_generation(eshares)

    for i in range(nclients):
        clients[i + 1].online_key_generation2(eshares[i + 1])
    all_y = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx].new_fl_step()
        user, y = clients[idx].online_encrypt()
        all_y[user] = y
    nclients_on = int(nclients - dropout)
    all_y = {idx: y for idx, y in all_y.items() if idx <= nclients_on}
    clients_on = server.online_encrypt(all_y)
    all_shares = {}
    for i in range(nclients_on):
        idx = i + 1
        user, shares = clients[idx].online_construct(clients_on)

        all_shares[user] = shares

    sum_protocol = server.online_construct(all_shares)

    sum_clear = clients[1].x
    from operator import add

    for i in range(1, nclients - dropout):
        sum_clear = list(map(add, sum_clear, clients[i + 1].x))

    assert sum_protocol == sum_clear
