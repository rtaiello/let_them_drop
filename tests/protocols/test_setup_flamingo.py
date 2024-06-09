from math import ceil, log2

from src.buildingblocks import Share
from src.protocols import FlamingoClient, FlamingoServer


def test_protocol():
    nclients = 10
    ndecryptors = -1
    dropout_rate = 0.0
    dimension = 1000
    inputsize = 16
    keysize = 256
    neighborood_size = ceil((nclients * dropout_rate) / log2(nclients))
    threshold = ceil(2 * nclients / 3)

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
    clients = {}
    for i in range(nclients):
        idx = i + 1
        clients[idx] = FlamingoClient(idx, 0, 0)
    server = FlamingoServer()
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
    all_shares_a = {}
    all_shares_b = {}
    all_commitments = {}
    for i in range(nclients):
        idx = i + 1
        user, shares_a, shares_b, commitments = clients[
            idx
        ].setup_send_shares_commits_sk()
        all_shares_a[user] = shares_a
        all_shares_b[user] = shares_b
        all_commitments[user] = commitments
    all_shares_a, all_shares_b, all_commitments = server.setup_send_shares_commits_sk(
        all_shares_a, all_shares_b, all_commitments
    )
    all_complaints = {}
    for i in range(nclients):
        idx = i + 1
        complaints = clients[idx].setup_receive_shares_commits_sk(
            all_shares_a[idx], all_shares_b[idx], all_commitments
        )
        all_complaints[idx] = complaints
    all_complaints = server.setup_accept_or_complain_sk(all_complaints)
    all_bcast_shares_a = {}
    all_bcast_shares_b = {}
    for i in range(nclients):
        idx = i + 1
        bcast_shares_a, bcast_shares_b = clients[idx].setup_accept_or_complain_sk(
            all_complaints
        )
        all_bcast_shares_a[idx] = bcast_shares_a
        all_bcast_shares_b[idx] = bcast_shares_b
    all_bcast_shares_a, all_bcast_shares_b = server.setup_forward_shares(
        all_bcast_shares_a, all_bcast_shares_b
    )
    all_qual = {}
    for i in range(nclients):
        idx = i + 1
        qual = clients[idx].setup_forward_shares(all_bcast_shares_a, all_bcast_shares_b)
        all_qual[idx] = qual
    all_qual = server.setup_broadcast_qual(all_qual)
    shares_a = []
    for i in range(nclients):
        idx = i + 1
        share = clients[idx].setup_create_sk(all_qual)
        field = FlamingoClient.teg.ss.Field
        shares_a.append(Share(idx, field(share)))
    all_commitments = {}
    for i in range(nclients):
        idx = i + 1
        commitments = clients[idx].setup_send_commits_pk()
        all_commitments[idx] = commitments
    all_commitments = server.setup_send_commits_pk(all_commitments)
    for i in range(nclients):
        idx = i + 1
        clients[idx].setup_receive_commits_pk(all_commitments)
    for i in range(nclients):
        idx = i + 1
        pk = clients[idx].setup_create_pk()

    lag_coeffs = FlamingoClient.teg.ss.lagrange(shares_a)
    sk = FlamingoClient.teg.ss.reconstruct(shares_a, threshold, lag_coeffs)
    base_point = FlamingoClient.g
    rencon_pk = sk * base_point

    assert rencon_pk == pk
