"""
Implementation of FTSA protocol https://dl.acm.org/doi/abs/10.1145/3564625.3568135
"""

from collections import defaultdict
from math import ceil, factorial

from gmpy2 import mpz

from ...buildingblocks import PRG, SSS, TJLS, VES, ServerKeyJL, sub_vectors


class ServerFTSA:
    dimension: int = 1000
    value_size: int = 16
    nclients: int = 10
    key_size: int = 2048
    threshold: int = ceil(2 * nclients / 3)
    ve: VES = VES(key_size // 2, nclients, value_size, dimension)
    tjl: TJLS = TJLS(threshold, nclients, ve)
    pp, _, _ = tjl.setup(key_size)
    prg: PRG = PRG(dimension, value_size)
    ss: SSS = SSS(PRG.security)

    def __init__(self) -> None:
        """
        Initializes the ServerFTSA object.

        Initializes various attributes for the FTSA server.
        """
        super().__init__()
        self.step: int = 0
        self.key: ServerKeyJL = ServerKeyJL(ServerFTSA.pp, mpz(0))
        self.clients: list = []
        self.clients_on: list = []
        self.y: list = []
        self.delta: int = 1

    @staticmethod
    def set_scenario(
        dimension: int, valuesize: int, keysize: int, threshold: int, nclients: int, pp
    ) -> None:
        """
        Set the scenario parameters for the FTSA server.

        Args:
            dimension (int): The dimension of the scenario.
            valuesize (int): The size of values.
            keysize (int): The size of the key.
            threshold (int): The threshold for FTSA.
            nclients (int): The number of clients.
            pp: Placeholder for the protocol parameters.
        """
        ServerFTSA.dimension = dimension
        ServerFTSA.value_size = valuesize
        ServerFTSA.nclients = nclients
        ServerFTSA.key_size = keysize
        ServerFTSA.threshold = threshold
        ServerFTSA.ve = VES(keysize // 2, nclients, valuesize, dimension)
        ServerFTSA.tjl = TJLS(threshold, nclients, ServerFTSA.ve)
        ServerFTSA.tjl.setup(keysize)
        ServerFTSA.pp = pp
        ServerFTSA.prg = PRG(dimension, valuesize)
        ServerFTSA.ss = SSS(PRG.security)

    def new_fl_step(self) -> None:
        """
        Advance to the next federated learning step.

        Resets various attributes for the next federated learning step.
        """
        self.step += 1
        self.clients_on = []
        self.y = []
        self.delta = 1

    def setup_register(self, alldhpkc, alldhpks):
        """
        Register and verify clients' public keys.

        Args:
            alldhpkc: Placeholder for clients' DH public keys.
            alldhpks: Placeholder for clients' DH private keys.

        Returns:
            Tuple: A tuple containing alldhpkc and alldhpks after verification.
        """
        assert alldhpkc.keys() == alldhpks.keys()
        assert len(alldhpkc.keys()) >= ServerFTSA.threshold

        return alldhpkc, alldhpks

    def setup_keysetup(self, all_e_messages):
        """
        Set up encryption keys based on clients' messages.

        Args:
            all_e_messages: Placeholder for encrypted messages from clients.

        Returns:
            dict: A dictionary containing encryption messages for further processing.
        """
        assert len(all_e_messages) >= ServerFTSA.threshold

        e_messages = defaultdict(dict)
        for user in all_e_messages:
            self.clients.append(user)
            for vuser in all_e_messages[user]:
                e_messages[vuser][user] = all_e_messages[user][vuser]

        self.delta = 1
        return e_messages

    def online_encrypt(self, all_e_shares, all_y):
        """
        Encrypt clients' shares online.

        Args:
            all_e_shares: Placeholder for encrypted shares from clients.
            all_y: Placeholder for clients' data.

        Returns:
            dict: A dictionary containing encrypted shares for further processing.
        """
        assert len(all_e_shares) >= ServerFTSA.threshold

        e_shares = defaultdict(dict)
        for user in all_e_shares:
            self.clients_on.append(user)
            for vuser in all_e_shares[user]:
                e_shares[vuser][user] = all_e_shares[user][vuser]

        self.y = list(all_y.values())

        return e_shares

    def online_construct(self, allbshares, y_zero_shares=None):
        """
        Construct shares online.

        Args:
            allbshares: Placeholder for shares from clients.
            y_zero_shares: Placeholder for shares of y_zero.

        Returns:
            dict: A dictionary containing the constructed aggregated value.
        """
        assert len(allbshares) >= ServerFTSA.threshold

        b_shares = defaultdict(list)
        for user in allbshares:
            for vuser in allbshares[user]:
                b_shares[vuser].append(allbshares[user][vuser])

        lag_coeffs = []
        b_mask = {}
        b_mask_vector = defaultdict(list)
        for vuser in b_shares:
            assert len(b_shares[vuser]) >= ServerFTSA.threshold
            if not lag_coeffs:
                lag_coeffs = ServerFTSA.ss.lagrange(b_shares[vuser])
            b_mask[vuser] = ServerFTSA.ss.reconstruct(
                b_shares[vuser], ServerFTSA.threshold, lag_coeffs
            )
            b_mask_vector[vuser] = ServerFTSA.prg.eval_vector(b_mask[vuser])
        y_zero_shares = [y for y in y_zero_shares if y]
        if y_zero_shares:
            assert len(y_zero_shares) >= ServerFTSA.threshold
            y_zero = ServerFTSA.tjl.share_combine(
                ServerFTSA.pp, y_zero_shares, self.threshold
            )
        else:
            y_zero = None

        x_masked = ServerFTSA.tjl.agg(
            ServerFTSA.pp, self.key, self.step, self.y, y_zero
        )

        for user in b_mask_vector:
            x_masked = sub_vectors(
                x_masked, b_mask_vector[user], 2 ** (ServerFTSA.ve.elementsize)
            )

        aggregated = x_masked
        return aggregated
