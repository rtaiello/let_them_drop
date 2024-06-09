from collections import defaultdict
from math import ceil
from typing import Dict, List

from gmpy2 import mpz

from ...buildingblocks import JLS, TJLS, VES, ServerKeyJL


class EagleServer:
    """
    EagleServer class for performing operations in a Eagle scheme.

    Attributes:
        dimension (int): The dimension of the scheme.
        valuesize (int): The size of the values.
        nclients (int): The number of clients.
        clients (List[int]): List of client IDs.
        threshold (int): The threshold value for the scheme.
        key_size_tjl (int): The key size for TJL.
        tjl (TJLS): An instance of the TJL class.
        pp_tjl: Parameters for TJL setup.
        key_size_jl (int): The key size for JL.
        ve (VES): An instance of the VES class.
        jl (JLS): An instance of the JLS class.
        pp_jl: Parameters for JL setup.
        step (int): A counter for the current step.
        key (ServerKeyJL): The server key for JL.
        clients (List[int]): List of clients involved in the current step.
        clients_on (List[int]): List of clients with online data available.
        y_key (List): List of Y key values.
        y (List): List of Y values.
        delta (int): A delta value.
    """

    dimension = 1000
    valuesize = 16
    nclients = 10
    clients = [i + 1 for i in range(nclients)]
    threshold = ceil(2 * nclients / 3)
    key_size_tjl = 4096
    tjl = TJLS(threshold, nclients)
    pp_tjl = tjl.setup(key_size_tjl)

    key_size_jl = 2048
    ve = VES(key_size_jl // 2, nclients, valuesize, dimension)
    jl = JLS(nclients, ve)
    pp_jl, _, _ = jl.setup(key_size_jl)

    def __init__(self) -> None:
        """
        Initialize the EagleServer object.

        Initializes the step counter, key, client lists, Y key and Y values, and delta.
        """
        super().__init__()
        self.step = 0
        self.key = ServerKeyJL(EagleServer.pp_tjl, mpz(0))
        self.clients = []
        self.clients_on = []
        self.y_key = []
        self.y = []
        self.delta = 1

    @staticmethod
    def set_scenario(
        dimension: int,
        value_size: int,
        key_size_tjl: int,
        key_size_jl: int,
        threshold: int,
        nclients: int,
        pp_tjl: any,  # Replace 'any' with the actual type of pp_tjl.
        pp_jl: any,  # Replace 'any' with the actual type of pp_jl.
        ndecryptors: int,
    ) -> None:
        """
        Set the scenario parameters for the EagleServer.

        Args:
            dimension (int): The dimension of the scheme.
            value_size (int): The size of the values.
            key_size_tjl (int): The key size for TJL.
            key_size_jl (int): The key size for JL.
            threshold (int): The threshold value for the scheme.
            nclients (int): The number of clients.
            pp_tjl: Parameters for TJL setup.
            pp_jl: Parameters for JL setup.
            ndecryptors (int): The number of decryptors.

        Raises:
            AssertionError: If ndecryptors is less than -1.
        """
        EagleServer.dimension = dimension
        EagleServer.valuesize = value_size
        EagleServer.nclients = nclients
        EagleServer.threshold = threshold

        EagleServer.key_size_tjl = key_size_tjl
        EagleServer.ndecryptors = ndecryptors
        if ndecryptors != -1:
            EagleServer.tjl = TJLS(threshold, ndecryptors)
            EagleServer.decryptors = [i + 1 for i in range(ndecryptors)]
        else:
            EagleServer.tjl = TJLS(threshold, nclients)
        EagleServer.tjl.setup(key_size_tjl)
        EagleServer.pp_tjl = pp_tjl

        EagleServer.key_size_jl = key_size_jl
        EagleServer.ve = VES(key_size_jl // 2, nclients, value_size, dimension)
        EagleServer.jl = JLS(nclients, EagleServer.ve)
        EagleServer.jl.setup(key_size_jl)
        EagleServer.pp_jl = pp_jl

    def new_fl_step(self) -> None:
        """
        Start a new federated learning step.

        Resets client lists, Y key and Y values, and delta.
        """
        self.step += 1
        self.clients = []
        self.y_key = []
        self.y = []
        self.delta = 1

    def setup_register(self, alldhpkc: Dict) -> Dict:
        """
        Setup the registration with DH public keys.

        Args:
            alldhpkc (Dict): A dictionary of DH public keys.

        Returns:
            Dict: The DH public keys.

        Raises:
            AssertionError: If the number of DH public keys is less than the threshold.
        """
        assert len(alldhpkc.keys()) >= EagleServer.threshold
        return alldhpkc

    def setup_keysetup(self, allekshares: Dict) -> Dict:
        """
        Setup the key generation with key shares.

        Args:
            allekshares (Dict): A dictionary of key shares.

        Returns:
            Dict: The aggregated key shares.

        Raises:
            AssertionError: If the number of key shares is less than the threshold.
        """
        assert len(allekshares) >= EagleServer.threshold

        ekshares = defaultdict(dict)
        for user in allekshares:
            self.clients.append(user)
            for vuser in allekshares[user]:
                ekshares[vuser][user] = allekshares[user][vuser]

        return ekshares

    def online_encrypt(self, all_y_key: Dict, all_y: Dict) -> List[int]:
        """
        Perform online encryption with Y keys and Y values.

        Args:
            all_y_key (Dict): A dictionary of Y keys.
            all_y (Dict): A dictionary of Y values.

        Returns:
            List[int]: List of clients with online data available.

        Raises:
            AssertionError: If the number of Y keys or Y values is less than the threshold.
        """
        assert len(all_y_key) >= EagleServer.threshold
        assert len(all_y) >= EagleServer.threshold

        self.y_key = list(all_y_key.values())
        self.y = list(all_y.values())
        self.clients_on = list(all_y_key.keys())
        return self.clients_on

    def online_construct(self, y_zero_shares: Dict) -> any:
        """
        Reconstruct the temporary JL Server key with TJL and complete the aggregation with JL.

        Args:
            y_zero_shares (Dict): A dictionary of Y zero shares.

        Returns:
            any: The aggregated result.

        Raises:
            AssertionError: If the number of Y zero shares is less than the threshold.
        """
        assert len(y_zero_shares) >= EagleServer.threshold
        y_zero_shares = list(y_zero_shares.values())
        y_zero_shares = y_zero_shares[: EagleServer.threshold]
        y_zero = EagleServer.tjl.share_combine(
            EagleServer.pp_tjl, y_zero_shares, EagleServer.threshold
        )
        aggregated_key = EagleServer.tjl.agg(
            EagleServer.pp_tjl, self.key, self.step, self.y_key, y_zero
        )
        aggregated_key = ServerKeyJL(EagleServer.pp_jl, -aggregated_key)
        aggregated = EagleServer.jl.agg(
            EagleServer.pp_jl, aggregated_key, 0, self.y
        )
        return aggregated
