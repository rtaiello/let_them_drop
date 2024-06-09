import random
from collections import defaultdict
from operator import add

import pytest

from src.buildingblocks import TD_TJLS, VES


@pytest.mark.parametrize("threshold, nusers, drops", [(3, 5, 0), (3, 5, 1), (3, 5, 2)])
def test_scheme(threshold, nusers, drops):
    tau = 412
    keysize = 2048
    dimension = 10
    inputsize = 16
    ve = VES(keysize // 2, nusers, inputsize, dimension)
    td_tjl_instance = TD_TJLS(threshold, nusers, ve)
    users = list(range(1, nusers + 1))
    alive = list(range(1, nusers - drops + 1))
    dropped = list(range(nusers - drops + 1, nusers + 1))

    pp, server_key, users_key = td_tjl_instance.setup(keysize)

    shares = {}
    for u in users:
        shares[u] = td_tjl_instance.sk_share(users_key[u - 1])

    users_shares = defaultdict(dict)
    for v in users:
        for u in users:
            users_shares[u][v] = shares[v][u - 1]
    xs = []
    for _ in range(nusers):
        x = []
        for _ in range(dimension):
            x.append(random.randint(0, 1000))
        xs.append(x)
    # encrypt all vectors
    ys = []
    for u in alive:
        y = td_tjl_instance.protect(pp, users_key[u - 1], tau, xs[u - 1])
        ys.append(y)
    if dropped != []:
        yzero_shares = {}
        for u in alive:
            dropped_users_shares = []
            for v in dropped:
                dropped_users_shares.append(users_shares[u][v])
            yzero_shares[u] = td_tjl_instance.share_protect(
                pp, dropped_users_shares, tau
            )

    if dropped != []:
        yzero_shares = list(yzero_shares.values())[:threshold]
        yzero = td_tjl_instance.share_combine(
            pp, yzero_shares, td_tjl_instance.threshold
        )
    else:
        yzero = None
    sum_protocol = td_tjl_instance.agg(pp, server_key, tau, ys, yzero)

    sum_clear = xs[0]
    for x in xs[1 : nusers - drops]:
        sum_clear = list(map(add, sum_clear, x))
    assert sum_clear == sum_protocol


if __name__ == "__main__":
    pytest.main()
