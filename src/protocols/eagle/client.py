from math import ceil, log2
from random import randint
from typing import Any, Dict, List, Tuple

import gmpy2
from gmpy2 import mpz

from ...buildingblocks import JLS, KAS, TJLS, VES
from ...buildingblocks import EncryptionKey as AESKEY
from ...buildingblocks import IShare, UserKeyJL


class EagleClient:
    """
    EagleClient class for performing operations in a Eagle scheme.

    Attributes:
        dimension (int): The dimension of the scheme.
        value_size (int): The size of the values.
        nclients (int): The number of clients.
        threshold (int): The threshold value for the scheme.
        clients (List[int]): List of client IDs.
        key_size_tjl (int): The key size for TJL.
        tjl (TJLS): An instance of the TJL class.
        pp_td_tjl: Parameters for TJL setup.
        key_size_jl (int): The key size for JL.
        ve (VES): An instance of the VES class.
        jl (JLS): An instance of the JLS class.
    """

    dimension = 1000
    value_size = 16
    nclients = 10
    threshold = ceil(2 * nclients / 3)
    clients = [i + 1 for i in range(nclients)]

    key_size_tjl = 4096
    tjl = TJLS(threshold, nclients)
    pp_td_tjl = tjl.setup(key_size_tjl)

    key_size_jl = 2048
    ve = VES(key_size_jl // 2, nclients, value_size, dimension)
    jl = JLS(nclients, ve)

    def __init__(self, user: int) -> None:
        """
        Initialize the EagleClient object.

        Args:
            user (int): The user ID.

        Initializes the user ID, step counter, key, channel keys, client lists, key shares, KAS channel,
        X key, and X values.
        """
        super().__init__()
        self.user = user
        self.step = 0
        self.key = UserKeyJL(EagleClient.pp_td_tjl, mpz(0))
        self.channel_keys = {}
        self.clients = []
        self.clients_on = []
        self.key_shares = {}
        self.ka_channel = KAS()
        self.x_key = []
        self.x = []

    @staticmethod
    def set_scenario(
        dimension: int,
        valuesize: int,
        keysize_tjl: int,
        keysize_jl: int,
        threshold: int,
        nclients: int,
        pp_td_tjl: Any,  # Replace 'Any' with the actual type of pp_td_tjl.
        pp_jl: Any,  # Replace 'Any' with the actual type of pp_jl.
        ndecryptors: int,
    ) -> None:
        """
        Set the scenario parameters for the EagleClient.

        Args:
            dimension (int): The dimension of the scheme.
            valuesize (int): The size of the values.
            keysize_tjl (int): The key size for TJL.
            keysize_jl (int): The key size for JL.
            threshold (int): The threshold value for the scheme.
            nclients (int): The number of clients.
            pp_td_tjl: Parameters for TJL setup.
            pp_jl: Parameters for JL setup.
            ndecryptors (int): The number of decryptors.

        Raises:
            AssertionError: If ndecryptors is less than -1.
        """
        EagleClient.dimension = dimension
        EagleClient.value_size = valuesize
        EagleClient.nclients = nclients
        EagleClient.threshold = threshold
        EagleClient.clients = [i + 1 for i in range(nclients)]

        EagleClient.key_size_tjl = keysize_tjl
        if ndecryptors != -1:
            EagleClient.tjl = TJLS(threshold, ndecryptors)
            EagleClient.decryptors = [i + 1 for i in range(ndecryptors)]
        else:
            EagleClient.tjl = TJLS(threshold, nclients)
            EagleClient.decryptors = None

        EagleClient.tjl.setup(keysize_tjl)
        EagleClient.pp_td_tjl = pp_td_tjl

        EagleClient.key_size_jl = keysize_jl
        EagleClient.ve = VES(keysize_jl // 2, nclients, valuesize, dimension)
        EagleClient.jl = JLS(nclients, EagleClient.ve)
        EagleClient.jl.setup(keysize_jl)
        EagleClient.pp_jl = pp_jl

    def new_fl_step(self) -> None:
        """
        Start a new federated learning step.

        Resets client lists, key shares, and generates X key and X values.
        """
        self.step += 1
        self.clients_on = []
        key_size_jl = ceil(
            EagleClient.key_size_jl - log2(EagleClient.nclients) - 1
        )
        x_key = randint(0, 2**key_size_jl)
        self.x_key = UserKeyJL(EagleClient.pp_jl, x_key)
        self.x = [1 for _ in range(EagleClient.dimension)]

    def setup_register(self) -> Tuple[int, Any]:
        """
        Setup the registration with a KAS channel.

        Returns:
            Tuple[int, Any]: User ID and KAS public key.
        """
        self.ka_channel.generate()
        self.clients.append(self.user)
        return self.user, self.ka_channel.pk

    def setup_keysetup(self, all_dh_pkc: Dict) -> Tuple[int, Dict[int, Any]]:
        """
        Setup the key generation with DH public keys.

        Args:
            all_dh_pkc (Dict): A dictionary of DH public keys.

        Returns:
            Tuple[int, Dict[int, Any]]: User ID and encrypted key shares.

        Raises:
            AssertionError: If the number of DH public keys is less than the threshold.
        """
        assert len(all_dh_pkc.keys()) >= self.threshold
        for vuser in all_dh_pkc:
            if vuser == self.user:
                continue
            self.clients.append(vuser)
            self.channel_keys[vuser] = self.ka_channel.agree(all_dh_pkc[vuser])

        key = mpz(randint(0, 2 * EagleClient.key_size_tjl - 1))
        self.key = UserKeyJL(EagleClient.pp_td_tjl, key)
        self.key.s = -self.key.s
        shares = EagleClient.tjl.sk_share(self.key)
        self.key.s = -self.key.s
        e_shares = {}
        for share in shares:
            vuser = share.idx
            if self.user == vuser:
                self.key_shares[self.user] = share
                continue
            key = AESKEY(self.channel_keys[vuser])
            message = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + gmpy2.to_binary(share.value)
            )
            e = key.encrypt(message)
            e_shares[vuser] = e

        return self.user, e_shares

    def setup_keysetup2(self, e_shares: Dict) -> None:
        """
        Setup the key generation (part 2) with encrypted key shares.

        Args:
            e_shares (Dict): A dictionary of encrypted key shares.

        Raises:
            AssertionError: If the number of encrypted key shares is less than the threshold.
        """
        assert len(e_shares) >= EagleClient.threshold
        for vuser in e_shares:
            key = AESKEY(self.channel_keys[vuser])
            message = key.decrypt(e_shares[vuser])
            u = int.from_bytes(message[:2], "big")
            v = int.from_bytes(message[2:4], "big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            share = gmpy2.from_binary(message[4:])
            self.key_shares[vuser] = IShare(self.user, share)

    def online_encrypt(self) -> Tuple[int, Any, Any]:
        """
        Encrypt the temporary JL user key with TJL, and encrypt the private input x with JL.

        Returns:
            Tuple[int, Any, Any]: User ID, encrypted Y key, and encrypted Y values.
        """
        y_key = EagleClient.tjl.protect(
            pp=EagleClient.pp_td_tjl,
            x_u_tau=self.x_key.s,
            sk_u=self.key,
            tau=self.step,
        )
        y = EagleClient.jl.protect(
            pp=EagleClient.pp_jl, x_u_tau=self.x, sk_u=self.x_key, tau=0
        )
        return self.user, y_key, y

    def online_construct(self, clients_on: List[int]) -> Tuple[int, Any]:
        """
        Reconstruct the scalar zero value to reconstruct the temporary JL server key.

        Args:
            clients_on (List[int]): List of clients with online data available.

        Returns:
            Tuple[int, Any]: User ID and reconstructed Y zero share key.

        Raises:
            AssertionError: If the number of online clients is less than the threshold.
        """
        assert len(clients_on) >= self.threshold
        self.clients_on = clients_on
        on_shares = []
        y_zero_share_key = None
        for vuser in self.clients_on:
            on_shares.append(self.key_shares[vuser])
        y_zero_share_key = EagleClient.tjl.share_protect(
            EagleClient.pp_td_tjl, on_shares, self.step
        )
        return self.user, y_zero_share_key
