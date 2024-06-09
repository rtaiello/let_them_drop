"""
Implementation of the SecAgg protocol. (Reference: https://eprint.iacr.org/2017/281.pdf)
"""

import gc
from collections import defaultdict
from math import ceil, log2
from typing import DefaultDict, Dict, List, Tuple

from ...buildingblocks import KAS, PRG, SSS, add_vectors, sub_vectors


class SecAggServer(object):
    dimension: int = 1000
    value_size: int = 16
    nclients: int = 10
    expanded_value_size: int = value_size + ceil(log2(nclients))
    keysize: int = 256
    threshold: int = ceil(2 * nclients / 3)
    prg: PRG = PRG(dimension, value_size)
    ss_b_mask: SSS = SSS(PRG.security)
    ss_sk: SSS = SSS(keysize)

    def __init__(self) -> None:
        """
        Initialize the SecAggServer.
        """
        super().__init__()
        self.step: int = 1
        self.clients: List[int] = []
        self.clients_2: List[int] = []
        self.clients_3: List[int] = []
        self.clients_5: List[int] = []
        self.all_dh_pks: Dict[int, bytes] = {}
        self.all_y: Dict[int, List[int]] = {}

    @staticmethod
    def set_scenario(
        dimension: int, value_size: int, keysize: int, threshold: int, nclients: int
    ) -> None:
        """
        Set the scenario parameters for SecAggServer.

        Args:
            dimension (int): Dimension parameter.
            valuesize (int): Value size parameter.
            keysize (int): Key size parameter.
            threshold (int): Threshold parameter.
            nclients (int): Number of clients.
        """
        SecAggServer.dimension = dimension
        SecAggServer.value_size = value_size
        SecAggServer.nclients = nclients
        SecAggServer.expanded_value_size = value_size + ceil(log2(nclients))
        SecAggServer.keysize = keysize
        SecAggServer.threshold = threshold
        SecAggServer.prg = PRG(dimension, value_size)
        SecAggServer.ss_b_mask = SSS(PRG.security)
        SecAggServer.ss_sk = SSS(keysize)

    def new_fl_step(self) -> None:
        """
        Advance to a new federated learning step, resetting server state.
        """
        self.step += 1
        self.all_y = {}

    def advertise_keys(
        self, all_dh_pks: Dict[int, bytes], all_dh_pkc: Dict[int, bytes]
    ) -> Tuple[Dict[int, bytes], Dict[int, bytes]]:
        """
        Advertise keys received from clients.

        Args:
            all_dh_pks (Dict[int, bytes]): All Diffie-Hellman public keys.
            all_dh_pkc (Dict[int, bytes]): All Diffie-Hellman public keys (encrypted).

        Returns:
            Tuple[Dict[int, bytes], Dict[int, bytes]]: All Diffie-Hellman public keys and encrypted keys.
        """
        self.clients = list(all_dh_pkc.keys())
        assert all_dh_pkc.keys() == all_dh_pks.keys()
        self.all_dh_pks = all_dh_pks
        return all_dh_pks, all_dh_pkc

    def share_keys(
        self, all_eshares: Dict[int, Dict[int, bytes]]
    ) -> Dict[int, Dict[int, bytes]]:
        """
        Share keys received from clients.

        Args:
            all_eshares (Dict[int, Dict[int, bytes]]): Encrypted shares from clients.

        Returns:
            Dict[int, Dict[int, bytes]]: Encrypted shares.
        """
        self.clients_2 = list(all_eshares.keys())
        assert len(self.clients_2) >= SecAggServer.threshold
        e_shares: DefaultDict[int, Dict[int, bytes]] = defaultdict(dict)

        for user in all_eshares:
            for vuser in all_eshares[user]:
                e_shares[vuser][user] = all_eshares[user][vuser]

        return e_shares

    def masked_input_collection(self, all_y: Dict[int, List[int]]) -> List[int]:
        """
        Collect masked input data from clients.

        Args:
            all_y (Dict[int, List[int]]): Masked input data from clients.

        Returns:
            List[int]: List of client identifiers.
        """
        self.clients_3 = list(all_y.keys())
        assert len(self.clients_3) >= SecAggServer.threshold
        self.all_y = all_y
        return self.clients_3

    def unmasking(
        self,
        all_sk_shares: Dict[int, Dict[int, List[int]]],
        all_b_shares: Dict[int, Dict[int, List[int]]],
    ) -> List[int]:
        """
        Unmask the masked input data.

        Args:
            all_sk_shares (Dict[int, Dict[int, List[int]]]): Secret key shares.
            all_b_shares (Dict[int, Dict[int, List[int]]]): b_mask shares.

        Returns:
            List[int]: Result of unmasking.
        """
        self.clients_5 = list(all_b_shares.keys())
        assert len(self.clients_5) >= SecAggServer.threshold

        bshares: DefaultDict[int, List[List[int]]] = defaultdict(list)
        for user in all_b_shares:
            for vuser in all_b_shares[user]:
                bshares[vuser].append(all_b_shares[user][vuser])

        lag_coeffs = []
        b_mask_vector_result = [0] * SecAggServer.dimension

        for vuser in bshares:
            assert len(bshares[vuser]) >= SecAggServer.threshold
            if not lag_coeffs:
                lag_coeffs = SecAggServer.ss_b_mask.lagrange(bshares[vuser])
            b_mask = SecAggServer.ss_b_mask.reconstruct(
                bshares[vuser], SecAggServer.threshold, lag_coeffs
            )
            b_mask_vector = SecAggServer.prg.eval_vector(b_mask)

            b_mask_vector_result = add_vectors(
                b_mask_vector_result,
                b_mask_vector,
                2**SecAggServer.expanded_value_size,
            )
            del b_mask_vector
            gc.collect()

        sk_shares: DefaultDict[int, List[List[int]]] = defaultdict(list)
        for user in all_sk_shares:
            for vuser in all_sk_shares[user]:
                sk_shares[vuser].append(all_sk_shares[user][vuser])

        dh_key: Dict[int, KAS] = {}
        lag_coeffs = []

        for vuser in sk_shares:
            assert len(sk_shares[vuser]) >= SecAggServer.threshold
            if not lag_coeffs:
                lag_coeffs = SecAggServer.ss_sk.lagrange(sk_shares[vuser])
            k = SecAggServer.ss_sk.reconstruct(
                sk_shares[vuser], SecAggServer.threshold, lag_coeffs
            )
            k = int(k)
            dh_key[vuser] = KAS().generate_from_bytes(
                k.to_bytes(SecAggServer.keysize // 8, "big")
            )

        key_vector_result = [0] * SecAggServer.dimension
        for user in self.clients:
            if user in self.clients_3:
                continue
            key_vector = [0] * SecAggServer.dimension
            for vuser in self.all_dh_pks:
                if vuser == user:
                    continue
                sv = dh_key[user].agree(self.all_dh_pks[vuser])
                if vuser > user:
                    key_vector = sub_vectors(
                        key_vector,
                        SecAggServer.prg.eval_vector(sv),
                        2**SecAggServer.expanded_value_size,
                    )
                else:
                    key_vector = add_vectors(
                        key_vector,
                        SecAggServer.prg.eval_vector(sv),
                        2**SecAggServer.expanded_value_size,
                    )

            key_vector_result = add_vectors(
                key_vector_result, key_vector, 2**SecAggServer.expanded_value_size
            )
            del key_vector
            gc.collect()

        result = [0] * SecAggServer.dimension
        for user in self.all_y:
            result = add_vectors(
                result, self.all_y[user], 2**SecAggServer.expanded_value_size
            )

        result = add_vectors(
            result, key_vector_result, 2**SecAggServer.expanded_value_size
        )
        del key_vector_result
        gc.collect()
        result = sub_vectors(
            result, b_mask_vector_result, 2**SecAggServer.expanded_value_size
        )
        del b_mask_vector_result
        gc.collect()

        return result
