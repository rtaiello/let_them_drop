from math import ceil, log2

from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

root_seed = get_random_bytes(32)


def find_neighbors(current_iteration, num_clients, id, neighborhood_size):
    nonce = b"\x00\x00\x00\x00\x00\x00\x00\x00"

    neighbors_list = set()  # a set, instead of a list

    # compute PRF(root, iter_num), output a seed. can use AES
    prf = ChaCha20.new(key=root_seed, nonce=nonce)
    current_seed = prf.encrypt(current_iteration.to_bytes(32, "big"))

    # compute PRG(seed), a binary string
    prg = ChaCha20.new(key=current_seed, nonce=nonce)

    # compute number of bytes we need for a graph
    num_choose = ceil(log2(num_clients))  # number of neighbors I choose
    num_choose = num_choose * neighborhood_size

    bytes_per_client = ceil(log2(num_clients) / 8)
    segment_len = num_choose * bytes_per_client
    num_rand_bytes = segment_len * num_clients
    data = b"a" * num_rand_bytes
    graph_string = prg.encrypt(data)

    # find the segment for myself
    my_segment = graph_string[
        (id - 1) * segment_len : (id - 1) * (segment_len) + segment_len
    ]

    # define the number of bits within bytes_per_client that can be convert to int (neighbor's ID)
    bits_per_client = ceil(log2(num_clients))
    # default number of clients is power of two
    for i in range(num_choose):
        tmp = my_segment[i * bytes_per_client : i * bytes_per_client + bytes_per_client]
        tmp_neighbor = int.from_bytes(tmp, "big") & ((1 << bits_per_client) - 1)

        if tmp_neighbor == id - 1:
            # print("client", self.id, " random neighbor choice happened to be itself, skip")
            continue
        if tmp_neighbor in neighbors_list:
            # print("client", self.id, "already chose", tmp_neighbor, "skip")
            continue
        neighbors_list.add(tmp_neighbor)

    # now we have a list for who I chose
    # find my ID in the rest, see which segment I am in. add to neighbors_list
    for i in range(num_clients):
        if i == id - 1:
            continue
        seg = graph_string[i * segment_len : i * (segment_len) + segment_len]
        ls = parse_segment_to_list(seg, num_choose, bits_per_client, bytes_per_client)
        if id - 1 in ls:
            # add current segment owner into neighbors_list
            neighbors_list.add(i)  # add current segment owner into neighbors_list
        # add plus one to each neighbor, since we start from 0
    neighbors_list = {i + 1 for i in neighbors_list}
    return neighbors_list


def parse_segment_to_list(segment, num_choose, bits_per_client, bytes_per_client):
    cur_ls = set()
    # take a segment (byte string), parse it to a list
    for i in range(num_choose):
        cur_bytes = segment[
            i * bytes_per_client : i * bytes_per_client + bytes_per_client
        ]

        cur_no = int.from_bytes(cur_bytes, "big") & ((1 << bits_per_client) - 1)

        cur_ls.add(cur_no)

    return cur_ls


def commit(commit_msg, base_point):
    commitment = commit_msg._value * base_point
    return commitment


def verify_commit(share_a, share_b, coeff_commit, g, h, idx):
    if share_b:
        lhs = (g * share_a.value._value) + (h * share_b.value._value)
    else:
        lhs = g * share_a.value._value
    rhs = coeff_commit[0]
    for i in range(1, len(coeff_commit)):
        rhs = idx * rhs + coeff_commit[i]
    if lhs.x != rhs.x and lhs.y != rhs.y:
        return False
    else:
        return True
