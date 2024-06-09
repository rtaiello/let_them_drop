import time
from math import ceil
from operator import add
from typing import Dict

import pytest

from src.buildingblocks import JLS, TJLS
from src.protocols import EagleClient, EagleServer


@pytest.mark.parametrize(
    "ndecryptors, nclients, dropout_rate",
    [
        (10, 32, 0.0),
        (10, 32, 0.1),
        (10, 32, 0.3),
        (-1, 32, 0.0),
        (-1, 32, 0.1),
        (-1, 32, 0.3),
    ],
)
def test_protocol(ndecryptors, nclients, dropout_rate):
    dimension = 1000
    input_size = 16
    key_size_tjl = 4096
    key_size_jl = 2048
    threshold = (
        ceil(2 * ndecryptors / 3) if ndecryptors != -1 else ceil(2 * nclients / 3)
    )
    dropout = ceil(nclients * dropout_rate)
    pp_tjl, _, _ = TJLS(threshold, nclients).setup(key_size_tjl)
    pp_jl, _, _ = JLS(threshold, nclients).setup(key_size_jl)

    EagleClient.set_scenario(
        dimension,
        input_size,
        key_size_tjl,
        key_size_jl,
        threshold,
        nclients,
        pp_tjl,
        pp_jl,
        ndecryptors,
    )
    EagleServer.set_scenario(
        dimension,
        input_size,
        key_size_tjl,
        key_size_jl,
        threshold,
        nclients,
        pp_tjl,
        pp_jl,
        ndecryptors,
    )

    clients: Dict[int, EagleClient] = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx] = EagleClient(idx)

    server = EagleServer()

    all_pkc = {}
    for i in range(nclients):
        idx = i + 1
        user, pkc = clients[idx].setup_register()
        all_pkc[user] = pkc

    all_pkc = server.setup_register(all_pkc)

    all_e_shares = {}
    for i in range(nclients):
        idx = i + 1
        user, eshares = clients[idx].setup_keysetup(all_pkc)
        all_e_shares[user] = eshares

    all_e_shares = server.setup_keysetup(all_e_shares)

    for i in range(nclients):
        idx = i + 1
        if ndecryptors != -1 and idx <= ndecryptors:
            clients[idx].setup_keysetup2(all_e_shares[idx])
        if ndecryptors == -1:
            clients[idx].setup_keysetup2(all_e_shares[idx])

    all_y_key = {}
    all_y = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx].new_fl_step()
        user, y_key, y = clients[idx].online_encrypt()
        all_y_key[user] = y_key
        all_y[user] = y
    nclients_on = nclients - ceil(nclients * dropout_rate)

    all_y = {idx: y for idx, y in all_y.items() if idx <= nclients_on}
    all_y_key = {idx: y_key for idx, y_key in all_y_key.items() if idx <= nclients_on}
    all_shares = {}
    server.new_fl_step()
    clients_on = server.online_encrypt(
        all_y_key, all_y
    )  # server.online_encrypt(all_y_key, all_y)
    nclients_on = len(clients_on)
    for i in range(nclients_on):
        idx = i + 1
        if ndecryptors != -1 and idx <= ndecryptors:
            user, shares = clients[idx].online_construct(clients_on)
            all_shares[user] = shares
        if ndecryptors == -1:
            user, shares = clients[idx].online_construct(clients_on)
            all_shares[user] = shares
    sum_protocol = server.online_construct(all_shares)

    sum_clear = clients[1].x

    for i in range(1, nclients - dropout):
        sum_clear = list(map(add, sum_clear, clients[i + 1].x))

    assert sum_protocol == sum_clear
