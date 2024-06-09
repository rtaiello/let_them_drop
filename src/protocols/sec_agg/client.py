"""
Implementation of the SecAgg protocol. (Reference: https://eprint.iacr.org/2017/281.pdf)
"""

import gc
from math import ceil, log2
from random import randint
from typing import Dict, List, Tuple

import gmpy2

from ...buildingblocks import KAS, PRG, SSS
from ...buildingblocks import EncryptionKey as AESKEY
from ...buildingblocks import Share, add_vectors, sub_vectors


class SecAggClient(object):
    dimension: int = 1000
    value_size: int = 16
    nclients: int = 10
    expanded_value_size: int = value_size + ceil(log2(nclients))
    keysize: int = 256
    threshold: int = ceil(2 * nclients / 3)
    clients: List[int] = [i + 1 for i in range(nclients)]
    prg: PRG = PRG(dimension, value_size)
    ss_b_mask: SSS = SSS(PRG.security)
    ss_sk: SSS = SSS(keysize)

    def __init__(self, user: int) -> None:
        """
        Initialize the SecAggClient.

        Args:
            user (int): The user identifier.
        """
        super().__init__()
        self.user = user
        self.step = 1
        self.all_dh_pkc: Dict[int, bytes] = {}
        self.clients: List[int] = []
        self.clients_on: List[int] = []
        self.b_shares: Dict[int, Share] = {}
        self.sk_shares: Dict[int, Share] = {}
        self.ka_sk: KAS = KAS()
        self.ka_channel: KAS = KAS()
        self.b_mask: int = 0
        self.all_dh_pks: Dict[int, bytes] = {}

    @staticmethod
    def set_scenario(
        dimension: int, valuesize: int, keysize: int, threshold: int, nclients: int
    ) -> None:
        """
        Set the scenario parameters for SecAggClient.

        Args:
            dimension (int): Dimension parameter.
            valuesize (int): Value size parameter.
            keysize (int): Key size parameter.
            threshold (int): Threshold parameter.
            nclients (int): Number of clients.
        """
        SecAggClient.dimension = dimension
        SecAggClient.value_size = valuesize
        SecAggClient.nclients = nclients
        SecAggClient.expanded_value_size = valuesize + ceil(log2(nclients))
        SecAggClient.keysize = keysize
        SecAggClient.threshold = threshold
        SecAggClient.clients = [i + 1 for i in range(nclients)]
        SecAggClient.prg = PRG(dimension, valuesize)
        SecAggClient.ss_b_mask = SSS(PRG.security)
        SecAggClient.ss_sk = SSS(keysize)

    def new_fl_step(self) -> None:
        """
        Advance to a new federated learning step, resetting client state.
        """
        self.step += 1
        self.clients = []
        self.clients_on = []
        self.clients_on = []
        self.b_shares = {}
        self.sk_shares = {}
        self.b_shares = {}
        self.ka_sk = KAS()
        self.ka_channel = KAS()
        self.all_dh_pkc = {}
        self.b_mask = 0
        self.all_dh_pks = {}

    def advertise_keys(self) -> Tuple[int, bytes, bytes]:
        """
        Advertise keys to other clients.

        Returns:
            Tuple[int, bytes, bytes]: User identifier, secret key, and channel key.
        """
        self.ka_sk.generate()
        self.ka_channel.generate()
        self.clients.append(self.user)
        return self.user, self.ka_sk.pk, self.ka_channel.pk

    def share_keys(
        self, all_dh_pks: Dict[int, bytes], all_dh_pkc: Dict[int, bytes]
    ) -> Tuple[int, Dict[int, bytes]]:
        """
        Share keys with other clients.

        Args:
            all_dh_pks (Dict[int, bytes]): All Diffie-Hellman public keys.
            all_dh_pkc (Dict[int, bytes]): All Diffie-Hellman public keys (encrypted).

        Returns:
            Tuple[int, Dict[int, bytes]]: User identifier and encrypted shares.
        """
        assert all_dh_pkc.keys() == all_dh_pks.keys()
        assert len(all_dh_pkc.keys()) >= self.threshold

        self.clients = list(all_dh_pkc.keys())
        for vuser in all_dh_pkc:
            if vuser == self.user:
                continue
            self.all_dh_pkc[vuser] = self.ka_channel.agree(all_dh_pkc[vuser])
        self.b_mask = randint(0, 2 ** (SecAggClient.prg.security) - 1)

        b_shares = SecAggClient.ss_b_mask.share(
            self.b_mask, SecAggClient.threshold, SecAggClient.nclients
        )

        sk_shares = SecAggClient.ss_sk.share(
            self.ka_sk.get_sk_bytes(),
            SecAggClient.threshold,
            SecAggClient.nclients,
        )

        e_shares = {}
        for kshare, bshare in zip(sk_shares, b_shares):
            assert kshare.idx == bshare.idx
            vuser = kshare.idx
            if self.user == vuser:
                self.sk_shares[self.user] = kshare
                self.b_shares[self.user] = bshare
                continue
            key = AESKEY(self.all_dh_pkc[vuser])
            sharelen = len(gmpy2.to_binary(kshare.value._value))
            message = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + sharelen.to_bytes(2, "big")
                + gmpy2.to_binary(kshare.value._value)
                + gmpy2.to_binary(bshare.value._value)
            )
            e = key.encrypt(message)
            e_shares[vuser] = e

        self.all_dh_pks = all_dh_pks

        return self.user, e_shares

    def masked_input_collection(
        self, eshares: Dict[int, bytes]
    ) -> Tuple[int, List[int]]:
        """
        Collect masked input data from other clients.

        Args:
            eshares (Dict[int, bytes]): Encrypted shares from other clients.

        Returns:
            Tuple[int, List[int]]: User identifier and masked input data.
        """
        assert len(eshares) >= self.threshold
        self.e_shares = eshares
        key = [0] * SecAggClient.dimension
        for vuser in self.all_dh_pks:
            if vuser == self.user:
                continue
            sv = self.ka_sk.agree(self.all_dh_pks[vuser])
            if vuser > self.user:
                key = sub_vectors(
                    key,
                    SecAggClient.prg.eval_vector(sv),
                    2**SecAggClient.expanded_value_size,
                )
            else:
                key = add_vectors(
                    key,
                    SecAggClient.prg.eval_vector(sv),
                    2**SecAggClient.expanded_value_size,
                )
            del sv
        b_mask_vector = SecAggClient.prg.eval_vector(self.b_mask)

        x_masked = add_vectors(
            key, b_mask_vector, 2**SecAggClient.expanded_value_size
        )
        del key
        x = [1 for _ in range(SecAggClient.dimension)]
        y = add_vectors(x, x_masked, 2**SecAggClient.expanded_value_size)
        del x
        del x_masked
        gc.collect()
        return self.user, y

    def unmasking(
        self, clients_on: List[int]
    ) -> Tuple[int, Dict[int, Share], Dict[int, Share]]:
        """
        Unmask the masked input data.

        Args:
            clients_on (List[int]): List of clients involved in unmasking.

        Returns:
            Tuple[int, Dict[int, Share], Dict[int, Share]]: User identifier, secret key shares, and b_mask shares.
        """
        assert len(clients_on) >= SecAggClient.threshold
        self.clients_on = clients_on
        assert set(self.clients_on).issubset(set(self.clients))

        for vuser in self.e_shares:
            key = AESKEY(self.all_dh_pkc[vuser])
            message = key.decrypt(self.e_shares[vuser])
            u = int.from_bytes(message[:2], "big")
            v = int.from_bytes(message[2:4], "big")
            sharelen = int.from_bytes(message[4:6], "big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            kshare = gmpy2.from_binary(message[6 : sharelen + 6])
            bshare = gmpy2.from_binary(message[sharelen + 6 :])
            self.b_shares[vuser] = Share(
                self.user, SecAggClient.ss_b_mask.Field(bshare)
            )
            self.sk_shares[vuser] = Share(self.user, SecAggClient.ss_sk.Field(kshare))

        b_mask_shares = {}
        sk_shares = {}
        for vuser in self.clients:
            if vuser in self.clients_on:
                b_mask_shares[vuser] = self.b_shares[vuser]
            else:
                sk_shares[vuser] = self.sk_shares[vuser]

        return self.user, sk_shares, b_mask_shares
