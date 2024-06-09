"""
Implementation of FTSA protocol https://dl.acm.org/doi/abs/10.1145/3564625.3568135
"""

from math import ceil
from random import randint

import gmpy2

from ...buildingblocks import KAS, PRG, SSS, TJLS, VES
from ...buildingblocks import EncryptionKey as AESKEY
from ...buildingblocks import IShare, Share, UserKeyJL, add_vectors


class ClientFTSA:
    dimension: int = 1000
    value_size: int = 16
    nclients: int = 10
    key_size: int = 2048
    threshold: int = ceil(2 * nclients / 3)
    clients: list = [i + 1 for i in range(nclients)]
    ve: VES = VES(key_size // 2, nclients, value_size, dimension)
    tjl: TJLS = TJLS(threshold, nclients, ve)
    pp, _, _ = tjl.setup(key_size)
    prg: PRG = PRG(dimension, value_size)
    ss: SSS = SSS(PRG.security)

    def __init__(self, user: int) -> None:
        """
        Initializes the ClientFTSA object.

        Args:
            user (int): The user ID for this client.
        """
        super().__init__()
        self.user: int = user
        self.step: int = 0
        self.key: gmpy2.mpz = gmpy2.mpz(0)
        self.channel_keys: dict = {}
        self.clients: list = []
        self.clients_on: list = []
        self.b_shares: dict = {}
        self.sk_shares: dict = {}
        self.x: list = []
        self.ka_sk: KAS = KAS()
        self.ka_channel: KAS = KAS()

    @staticmethod
    def set_scenario(
        dimension: int,
        valuesize: int,
        keysize: int,
        threshold: int,
        nclients: int,
        publicparam,
    ) -> None:
        """
        Set the scenario parameters for the FTSA client.

        Args:
            dimension (int): The dimension of the scenario.
            valuesize (int): The size of values.
            keysize (int): The size of the key.
            threshold (int): The threshold for FTSA.
            nclients (int): The number of clients.
            publicparam: Placeholder for the public parameters.
        """
        ClientFTSA.dimension = dimension
        ClientFTSA.value_size = valuesize
        ClientFTSA.nclients = nclients
        ClientFTSA.key_size = keysize
        ClientFTSA.threshold = threshold
        ClientFTSA.clients = [i + 1 for i in range(nclients)]
        ClientFTSA.ve = VES(keysize // 2, nclients, valuesize, dimension)
        ClientFTSA.tjl = TJLS(threshold, nclients, ClientFTSA.ve)
        ClientFTSA.tjl.setup(keysize)
        ClientFTSA.pp = publicparam
        ClientFTSA.prg = PRG(dimension, valuesize)
        ClientFTSA.ss = SSS(PRG.security)

    def new_fl_step(self) -> None:
        """
        Advance to the next federated learning step.

        Resets various attributes for the next federated learning step.
        """
        self.step += 1
        self.clients_on = []
        self.b_shares = {}
        self.x = [1 for _ in range(ClientFTSA.dimension)]

    def setup_register(self) -> tuple:
        """
        Register and generate key pairs for the client.

        Returns:
            tuple: A tuple containing the user ID, the secret key, and the channel key.
        """
        self.ka_sk.generate()
        self.ka_channel.generate()
        self.clients.append(self.user)
        return self.user, self.ka_sk.pk, self.ka_channel.pk

    def setup_keysetup(self, all_dh_pks, all_dh_pkc) -> tuple:
        """
        Set up encryption keys based on clients' public keys.

        Args:
            all_dh_pks: Placeholder for clients' DH public keys.
            all_dh_pkc: Placeholder for clients' DH private keys.

        Returns:
            tuple: A tuple containing the user ID and encrypted messages for further processing.
        """
        assert all_dh_pkc.keys() == all_dh_pks.keys()
        assert len(all_dh_pkc.keys()) >= self.threshold

        for vuser in all_dh_pkc:
            if vuser == self.user:
                continue
            self.clients.append(vuser)
            self.channel_keys[vuser] = self.ka_channel.agree(all_dh_pkc[vuser])
            sv = self.ka_sk.agree(all_dh_pks[vuser], ClientFTSA.key_size)
            if vuser > self.user:
                self.key -= sv
            else:
                self.key += sv

        self.key = UserKeyJL(ClientFTSA.pp, self.key)
        shares = ClientFTSA.tjl.sk_share(self.key)
        e_messages = {}
        for share in shares:
            vuser = share.idx
            if self.user == vuser:
                self.sk_shares[self.user] = share
                continue
            key = AESKEY(self.channel_keys[vuser])
            message = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + gmpy2.to_binary(share.value)
            )
            e = key.encrypt(message)
            e_messages[vuser] = e
        return self.user, e_messages

    def setup_keysetup2(self, e_shares) -> None:
        """
        Set up encryption keys based on encrypted messages from other clients.

        Args:
            e_shares: Placeholder for encrypted shares from other clients.
        """
        assert len(e_shares) + 1 >= self.threshold
        for vuser in e_shares:
            key = AESKEY(self.channel_keys[vuser])
            message = key.decrypt(e_shares[vuser])
            u = int.from_bytes(message[:2], "big")
            v = int.from_bytes(message[2:4], "big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            share = gmpy2.from_binary(message[4:])
            self.sk_shares[vuser] = IShare(self.user, share)

    def online_encrypt(self) -> tuple:
        """
        Encrypt client's data online.

        Returns:
            tuple: A tuple containing the user ID, encrypted messages, and y value.
        """
        b_mask = randint(0, 2**PRG.security - 1)
        b_mask_vector = ClientFTSA.prg.eval_vector(b_mask)
        x_masked = add_vectors(self.x, b_mask_vector, 2 ** (ClientFTSA.ve.elementsize))
        y = ClientFTSA.tjl.protect(ClientFTSA.pp, self.key, self.step, x_masked)
        shares = ClientFTSA.ss.share(b_mask, self.threshold, self.nclients)
        e_messages = {}
        for share in shares:
            vuser = share.idx
            if self.user == vuser:
                self.b_shares[self.user] = share
                continue
            key = AESKEY(self.channel_keys[vuser])
            message = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + gmpy2.to_binary(share.value._value)
            )
            e = key.encrypt(message)
            e_messages[vuser] = e
        return self.user, e_messages, y

    def online_construct(self, e_shares) -> tuple:
        """
        Construct shares online.

        Args:
            e_shares: Placeholder for encrypted shares from other clients.

        Returns:
            tuple: A tuple containing the user ID, constructed shares, and y_zero share.
        """
        assert len(e_shares) + 1 >= self.threshold
        self.clients_on = [self.user]
        for vuser in e_shares:
            self.clients_on.append(vuser)
            key = AESKEY(self.channel_keys[vuser])
            message = key.decrypt(e_shares[vuser])
            u = int.from_bytes(message[:2], "big")
            v = int.from_bytes(message[2:4], "big")
            share = gmpy2.from_binary(message[4:])
            assert v == self.user and u == vuser, "invalid encrypted message"
            self.b_shares[vuser] = Share(self.user, ClientFTSA.ss.Field(share))
        drop_shares = []
        y_zero_share = None
        if self.clients != self.clients_on:
            for vuser in self.clients:
                if vuser in self.clients_on:
                    continue
                drop_shares.append(self.sk_shares[vuser])
            if drop_shares:
                y_zero_share = ClientFTSA.tjl.share_protect(
                    ClientFTSA.pp, drop_shares, self.step
                )
        return self.user, self.b_shares, y_zero_share
