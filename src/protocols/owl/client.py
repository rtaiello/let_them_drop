import random
from math import ceil, log2
from random import randint

import gmpy2
from gmpy2 import mpz

from ...buildingblocks import JLS, KAS, SSS, VES
from ...buildingblocks import EncryptionKey as AESKEY
from ...buildingblocks import Share, UserKeyJL


class OwlClient(object):
    """
    Asynchronous Fault-Tolerant Client for Secure Aggregation

    Attributes:
        dimension (int): The dimension of the data.
        value_size (int): The size of each value.
        nclients (int): The number of clients.
        key_size (int): The size of the cryptographic key.
        threshold (int): The threshold value for secure operations.
        clients (list): A list of client IDs.
        ve (VES): An instance of the VES class.
        jl (JLS): An instance of the JLS class.
        pp (tuple): A tuple containing parameters from JLS setup.
        ss (SSS): An instance of the SSS class.
    """

    dimension = 1000
    value_size = 16
    nclients = 10
    key_size = 2048
    threshold = ceil(2 * nclients / 3)
    clients = [i + 1 for i in range(nclients)]
    ve = VES(key_size // 2, nclients, value_size, dimension)
    jl = JLS(nclients, ve)
    pp, _, _ = jl.setup(key_size)
    ss = SSS(key_size)

    def __init__(self, user: int) -> None:
        """
        Initialize the AsyncFTSAClient instance.

        Args:
            user (int): The ID of the client.
        """
        super().__init__()
        self.shares = {}
        self.user = user
        self.step = 0
        self.key = None
        self.ckeys = {}
        self.clients = []
        self.clients_on = []
        self.key_shares = {}
        self.x = []
        self.ka_channel = KAS()

    @staticmethod
    def set_scenario(
        dimension: int,
        value_size: int,
        key_size: int,
        threshold: int,
        nclients: int,
        publicparam: tuple,
    ) -> None:
        """
        Set the scenario parameters for the client.

        Args:
            dimension (int): The dimension of the data.
            value_size (int): The size of each value.
            key_size (int): The size of the cryptographic key.
            threshold (int): The threshold value for secure operations.
            nclients (int): The number of clients.
            publicparam (tuple): A tuple containing parameters from JLS setup.

        Returns:
            None
        """
        OwlClient.dimension = dimension
        OwlClient.value_size = value_size
        OwlClient.nclients = nclients
        OwlClient.key_size = key_size
        OwlClient.threshold = threshold
        OwlClient.clients = [i + 1 for i in range(nclients)]
        OwlClient.VE = VES(key_size // 2, nclients, value_size, dimension)
        OwlClient.jl = JLS(nclients, OwlClient.VE)
        OwlClient.jl.setup(key_size)
        OwlClient.pp = publicparam
        OwlClient.ss = SSS(key_size)

    def new_fl_step(self) -> None:
        """
        Advance to the next federated learning step.

        Returns:
            None
        """
        self.step += 1
        self.clients_on = []
        self.x = [1 for _ in range(OwlClient.dimension)]

    def setup_register(self) -> tuple:
        """
        Register the client with the key agreement channel.

        Returns:
            tuple: A tuple containing the user ID and public key.
        """
        self.ka_channel.generate()
        self.clients.append(self.user)
        return self.user, self.ka_channel.pk

    def setup_keysetup(self, alldhpkc: dict) -> None:
        """
        Perform key setup and share keys with other clients.

        Args:
            alldhpkc (dict): A dictionary of Diffie-Hellman public keys for clients.

        Returns:
            None
        """
        assert len(alldhpkc.keys()) >= self.threshold
        for vuser in alldhpkc:
            if vuser == self.user:
                continue
            self.clients.append(vuser)
            self.ckeys[vuser] = self.ka_channel.agree(alldhpkc[vuser])

    def online_key_generation(self) -> tuple:
        """
        Perform online key generation for the client and share encrypted key shares.

        Returns:
            tuple: A tuple containing the user ID and encrypted key shares.
        """
        key_size_jl = ceil(
            OwlClient.key_size - log2(OwlClient.nclients) - 1
        )
        key = randint(0, 2**key_size_jl)
        self.key = UserKeyJL(OwlClient.pp, key)
        shares = OwlClient.ss.share(
            self.key.s, OwlClient.threshold, len(OwlClient.clients)
        )
        e_shares = {}
        for share in shares:
            vuser = share.idx
            if self.user == vuser:
                self.key_shares[self.user] = share
                continue
            key = AESKEY(self.ckeys[vuser])
            message = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + gmpy2.to_binary(share.value._value)
            )
            e = key.encrypt(message)
            e_shares[vuser] = e

        return self.user, e_shares

    def online_key_generation2(self, e_shares: dict) -> None:
        """
        Decrypt and process encrypted key shares received from other clients.

        Args:
            e_shares (dict): A dictionary of encrypted key shares.

        Returns:
            None
        """
        assert len(e_shares) + 1 >= self.threshold

        for vuser in e_shares:
            key = AESKEY(self.ckeys[vuser])
            message = key.decrypt(e_shares[vuser])
            u = int.from_bytes(message[:2], "big")
            v = int.from_bytes(message[2:4], "big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            share = gmpy2.from_binary(message[4:])
            self.key_shares[vuser] = Share(self.user, OwlClient.ss.Field(share))
        return

    def online_encrypt(self) -> tuple:
        """
        Perform online encryption for the client.

        Returns:
            tuple: A tuple containing the user ID and encrypted values.
        """
        y = OwlClient.jl.protect(
            pp=OwlClient.pp, x_u_tau=self.x, sk_u=self.key, tau=0
        )
        return self.user, y

    def online_construct(self, clients_on: list) -> tuple:
        """
        Perform online key construction and aggregation of encrypted values.

        Args:
            clients_on (list): A list of clients currently online.

        Returns:
            tuple: A tuple containing the user ID and aggregated encrypted values.
        """
        self.clients_on = clients_on
        nclients_on = len(clients_on)
        assert nclients_on + 1 >= self.threshold
        sk_0_share = self.key_shares[1]
        for vuser in range(2, nclients_on + 1):
            sk_0_share += self.key_shares[vuser]
        return self.user, sk_0_share
