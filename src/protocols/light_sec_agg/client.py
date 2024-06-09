"""
Implementation LightSecAgg protocol [1] https://arxiv.org/pdf/2109.14236.pdf.
"""
from math import ceil
from typing import Dict, List, Tuple

import gmpy2

from ...buildingblocks import KAS
from ...buildingblocks import EncryptionKey as AESKEY
from ...buildingblocks import add_vectors, create_mask, create_random_vector, get_field
from ...buildingblocks.lcc import LCC


class LightSecAggClient:
    dimension: int = 1000
    nclients: int = 10
    valuesize: int = 16
    key_size: int = 64
    threshold: int = ceil(2 * nclients / 3)
    clients: List[int] = [i + 1 for i in range(nclients)]
    lcc: LCC = LCC(key_size)

    def __init__(self, user: int) -> None:
        """
        Initialize a LightSecAggClient instance.

        Args:
            user (int): The user ID.

        Returns:
            None
        """
        super().__init__()
        self.user: int = user
        self.step: int = 0
        self.ckeys: Dict[int, gmpy2.mpz] = {}
        self.u: List[int] = []
        self.clients_on: List[int] = []
        self.key_shares: Dict[int, List[gmpy2.mpz]] = {}
        self.x: List[gmpy2.mpz] = []
        self.ka_c: KAS = KAS()
        self.local_mask: List[gmpy2.mpz] = []
        self.mask_shares: Dict[int, List[gmpy2.mpz]] = {}

    @staticmethod
    def set_scenario(
        dimension: int, valuesize: int, keysize: int, threshold: int, nclients: int
    ) -> None:
        """
        Set the scenario parameters for LightSecAggClient.

        Args:
            dimension (int): The dimension.
            valuesize (int): The valuesize.
            keysize (int): The keysize.
            threshold (int): The threshold.
            nclients (int): The number of clients.

        Returns:
            None
        """
        LightSecAggClient.dimension = dimension
        LightSecAggClient.valuesize = valuesize
        LightSecAggClient.nclients = nclients
        LightSecAggClient.threshold = threshold
        LightSecAggClient.nclients_target_on = threshold + 1
        LightSecAggClient.keysize = keysize
        LightSecAggClient.clients = [i + 1 for i in range(nclients)]
        LightSecAggClient.lcc = LCC(keysize)
        LightSecAggClient.field = get_field(keysize)

    def new_fl_step(self) -> None:
        """
        Initialize a new federated learning step.

        Returns:
            None
        """
        self.step += 1
        self.clients_on = []
        self.shares: Dict[int, List[gmpy2.mpz]] = {}
        self.x = create_random_vector(
            LightSecAggClient.keysize,
            LightSecAggClient.valuesize,
            LightSecAggClient.dimension,
        )

    def register(self) -> Tuple[int, gmpy2.mpz]:
        """
        Perform the setup registration step.

        Returns:
            Tuple[int, gmpy2.mpz]: A tuple containing the user ID and public key.
        """
        self.ka_c.generate()
        self.u.append(self.user)
        return self.user, self.ka_c.pk

    def key_setup(self, alldhpks: Dict[int, gmpy2.mpz]) -> None:
        """
        Perform the key setup step.

        Args:
            alldhpks (Dict[int, gmpy2.mpz]): A dictionary of all users' public keys.

        Returns:
            None
        """
        for vuser in alldhpks:
            if vuser == self.user:
                continue
            self.ckeys[vuser] = self.ka_c.agree(alldhpks[vuser])

    def share_local_mask(self) -> Tuple[int, Dict[int, bytes]]:
        """
        Share the local mask.

        Returns:
            Tuple[int, Dict[int, bytes]]: A tuple containing the user ID and encrypted shares.
        """
        self.local_mask = create_mask(
            LightSecAggClient.keysize, LightSecAggClient.dimension
        )
        encoded_mask_set = self.lcc.mask_encoding(
            LightSecAggClient.dimension,
            LightSecAggClient.nclients,
            LightSecAggClient.nclients_target_on,
            LightSecAggClient.threshold,
            self.local_mask,
        )
        e_shares: Dict[int, bytes] = {}
        for vuser, shares in encoded_mask_set.items():
            vuser = vuser + 1
            if vuser == self.user:
                self.mask_shares[vuser] = shares
                continue
            key = AESKEY(self.ckeys[vuser])
            binaries_shares = []
            for share in shares:
                share = gmpy2.to_binary(share._value)
                sharelen = len(share)
                binaries_shares.append(sharelen.to_bytes(2, "big"))
                binaries_shares.append(share)
            binaries_shares = b"".join(binaries_shares)
            message = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + binaries_shares
            )
            e = key.encrypt(message)
            e_shares[vuser] = e
        return self.user, e_shares

    def receive_local_masks(self, e_shares: Dict[int, bytes]) -> None:
        """
        Receive and decrypt the local masks.

        Args:
            e_shares (Dict[int, bytes]): A dictionary of encrypted shares.

        Returns:
            None
        """
        for vuser in e_shares.keys():
            if vuser == self.user:
                continue
            key = AESKEY(self.ckeys[vuser])
            message = key.decrypt(e_shares[vuser])
            u = int.from_bytes(message[:2], byteorder="big")
            v = int.from_bytes(message[2:4], byteorder="big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            i = 4
            decrypted_shares = []
            while i < len(message):
                share_len = int.from_bytes(message[i : i + 2], byteorder="big")
                share = message[i + 2 : i + 2 + share_len]
                decrypted_share = LightSecAggClient.field(gmpy2.from_binary(share))
                decrypted_shares.append(decrypted_share)
                i = i + 2 + share_len
            self.mask_shares[vuser] = decrypted_shares

    def online_encrypt(self) -> Tuple[int, List[gmpy2.mpz]]:
        """
        Perform online encryption.

        Returns:
            Tuple[int, List[gmpy2.mpz]]: A tuple containing the user ID and encrypted data.
        """
        y = add_vectors(self.x, self.local_mask)
        return self.user, y

    def one_shot_recovery(self, clients_on: List[int]) -> Tuple[int, List[gmpy2.mpz]]:
        """
        Perform one-shot recovery.

        Args:
            clients_on (List[int]): A list of client IDs.

        Returns:
            Tuple[int, List[gmpy2.mpz]]: A tuple containing the user ID and recovered data.
        """
        d_encoded = ceil(
            LightSecAggClient.dimension
            / (LightSecAggClient.nclients_target_on - LightSecAggClient.threshold)
        )
        sum_encoded_mask: List[gmpy2.mpz] = [LightSecAggClient.field(0)] * d_encoded
        for vuser in clients_on:
            for k in range(d_encoded):
                sum_encoded_mask[k] += self.mask_shares[vuser][k]
        return self.user, sum_encoded_mask
