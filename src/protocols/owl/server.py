from collections import defaultdict
from math import ceil

from gmpy2 import mpz

from ...buildingblocks import JLS, SSS, VES, ServerKeyJL


class OwlServer(object):
    """
    Asynchronous Fault-Tolerant Server for Secure Aggregation

    Attributes:
        dimension (int): The dimension of the data.
        valuesize (int): The size of each value.
        nclients (int): The number of clients.
        keysize (int): The size of the cryptographic key.
        threshold (int): The threshold value for secure operations.
        clients (list): A list of client IDs.
        ve (VES): An instance of the VES class.
        jl (JLS): An instance of the JLS class.
        pp (tuple): A tuple containing parameters from JLS setup.
        ss (SSS): An instance of the SSS class.
    """

    dimension = 1000
    valuesize = 16
    nclients = 10
    keysize = 2048
    threshold = ceil(2 * nclients / 3)
    clients = [i + 1 for i in range(nclients)]
    ve = VES(keysize // 2, nclients, valuesize, dimension)
    jl = JLS(nclients, ve)
    pp, _, _ = jl.setup(keysize)
    ss = SSS(keysize)

    def __init__(self) -> None:
        """
        Initialize the AsyncFTSAServer instance.
        """
        super().__init__()
        self.step = 0
        self.key = ServerKeyJL(OwlServer.pp, mpz(0))
        self.clients = []
        self.clients_on = []
        self.y = []

    @staticmethod
    def set_scenario(
        dimension: int,
        valuesize: int,
        keysize: int,
        threshold: int,
        nclients: int,
        pp: tuple,
    ) -> None:
        """
        Set the scenario parameters for the server.

        Args:
            dimension (int): The dimension of the data.
            valuesize (int): The size of each value.
            keysize (int): The size of the cryptographic key.
            threshold (int): The threshold value for secure operations.
            nclients (int): The number of clients.
            pp (tuple): A tuple containing parameters from JLS setup.

        Returns:
            None
        """
        OwlServer.dimension = dimension
        OwlServer.valuesize = valuesize
        OwlServer.nclients = nclients
        OwlServer.keysize = keysize
        OwlServer.threshold = threshold
        OwlServer.ve = VES(keysize // 2, nclients, valuesize, dimension)
        OwlServer.jl = JLS(nclients, OwlServer.ve)
        OwlServer.jl.setup(keysize)
        OwlServer.pp = pp
        OwlServer.ss = SSS(keysize)

    def new_fl_step(self) -> None:
        """
        Advance to the next federated learning step.

        Returns:
            None
        """
        self.step += 1
        self.clients_on = []
        self.y = []

    def setup_register(self, alldhpkc: dict) -> dict:
        """
        Register the Diffie-Hellman public keys of clients and perform threshold checks.

        Args:
            alldhpkc (dict): A dictionary of Diffie-Hellman public keys for clients.

        Returns:
            dict: The registered Diffie-Hellman public keys.
        """
        assert len(alldhpkc.keys()) >= OwlServer.threshold
        return alldhpkc

    def online_key_generation(self, all_e_shares: dict) -> dict:
        """
        Perform online key generation for clients and gather encrypted key shares.

        Args:
            all_e_shares (dict): A dictionary of encrypted key shares for clients.

        Returns:
            dict: Encrypted key shares for clients.
        """
        assert len(all_e_shares) >= OwlServer.threshold
        ekshares = defaultdict(dict)
        for user in all_e_shares:
            self.clients.append(user)
            for vuser in all_e_shares[user]:
                ekshares[vuser][user] = all_e_shares[user][vuser]
        return ekshares

    def online_encrypt(self, all_y: dict) -> list:
        """
        Perform online encryption for clients and gather encrypted values.

        Args:
            all_y (dict): A dictionary of values to be encrypted for clients.

        Returns:
            list: A list of clients currently online.
        """
        assert len(all_y) >= OwlServer.threshold
        self.y = list(all_y.values())
        self.clients_on = list(all_y.keys())
        return self.clients_on

    def online_construct(self, sk_0_shares: dict):
        """
        Perform online key construction and aggregation of encrypted values.

        Args:
            sk_0_shares (dict): A dictionary of shared secret key values.

        Returns:
            aggregated: The aggregated encrypted values.
        """
        sk_0_shares = list(sk_0_shares.values())
        assert len(sk_0_shares) >= OwlServer.threshold
        lag_coeffs = OwlServer.ss.lagrange(sk_0_shares)
        sk_0 = OwlServer.ss.reconstruct(
            sk_0_shares, OwlServer.threshold, lag_coeffs
        )
        self.key = ServerKeyJL(OwlServer.pp, -sk_0)
        aggregated = OwlServer.jl.agg(OwlServer.pp, self.key, 0, self.y)
        return aggregated
