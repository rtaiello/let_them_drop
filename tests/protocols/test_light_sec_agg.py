from math import ceil, floor

import pytest

from src.buildingblocks import add_vectors
from src.protocols import LightSecAggClient, LightSecAggServer


@pytest.mark.parametrize("nclients, dropout_rate", [(32, 0.0), (32, 0.1), (32, 0.3)])
def test_protocol(nclients, dropout_rate):
    dropout = floor(nclients * dropout_rate)
    threshold = ceil(2 * nclients / 3)
    nclients_on = int(nclients - dropout)
    dimension = 100
    valuesize = 16
    keysize = 256
    LightSecAggClient.set_scenario(dimension, valuesize, keysize, threshold, nclients)
    LightSecAggServer.set_scenario(dimension, valuesize, keysize, threshold, nclients)
    clients = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx] = LightSecAggClient(idx)
    server = LightSecAggServer()
    allpks = {}
    for i in range(nclients):
        idx = i + 1
        user, pk = clients[idx].register()
        allpks[user] = pk

    server.setup_register(allpks)

    for i in range(nclients):
        idx = i + 1
        clients[idx].key_setup(allpks)
    for i in range(nclients):
        idx = i + 1
        clients[idx].new_fl_step()

    alleshares = {}
    for i in range(nclients):
        idx = i + 1
        user, eshares = clients[idx].share_local_mask()
        alleshares[user] = eshares

    eshares = server.distribute_local_masks(alleshares)

    for i in range(nclients):
        idx = i + 1
        clients[idx].receive_local_masks(eshares[idx])

    all_y = {}
    for i in range(nclients):
        idx = i + 1
        user, y = clients[idx].online_encrypt()
        all_y[user] = y

    nclients_on = int(nclients - dropout)
    all_y = {idx: y for idx, y in all_y.items() if idx <= nclients_on}
    clients_on = server.online_encrypt(all_y)

    all_sum_encoded_mask = {}
    for i in range(nclients_on):
        idx = i + 1
        user, sum_encoded_mask = clients[idx].one_shot_recovery(clients_on)
        all_sum_encoded_mask[user] = sum_encoded_mask

    sum_protocol = server.aggregate(all_sum_encoded_mask)
    sum_clear = clients[1].x
    for i in range(2, nclients_on + 1):
        sum_clear = add_vectors(sum_clear, clients[i].x)
    assert sum_protocol == sum_clear
