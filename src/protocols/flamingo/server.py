"""
Implementation of Flamingo protocol https://eprint.iacr.org/2023/486.pdf.
"""
import gc
from collections import defaultdict
from math import ceil, log2
from typing import Any, Dict, List, Tuple, Union


from ...buildingblocks.el_gamal.teg import TElGamal
from ...buildingblocks.prf import PRF
from ...buildingblocks.prg import PRG
from ...buildingblocks.ss.shamir_ss import SSS
from ...buildingblocks.utils import add_vectors, sub_vectors
from ..flamingo.utils import find_neighbors


class FlamingoServer(object):
    dimension: int = 1000
    value_size: int = 16
    nclients: int = 10
    expanded_value_size: int = value_size + ceil(log2(nclients))
    keysize: int = 256
    threshold: int = ceil(2 * nclients / 3)

    def __init__(self) -> None:
        """
        Initialize a FlamingoServer instance.
        """
        super().__init__()
        self.step: int = 1
        self.clients: List[int] = []
        self.clients_on: List[int] = []
        self.clients_off: List[int] = []
        self.all_dh_pks: Dict[int, Any] = {}
        self.all_e_shares: Dict[int, Dict[int, Any]] = defaultdict(dict)
        self.all_e_messages: Dict[Tuple[int, int], Any] = defaultdict(dict)
        self.all_y: Dict[int, List[int]] = {}
        self.decryptors: List[int] = [i + 1 for i in range(60)]
        self.key_shares: Dict[int, Any] = {}

    @staticmethod
    def set_scenario(
        dimension: int,
        valuesize: int,
        keysize: int,
        threshold: int,
        nclients: int,
        neighborood_size: int,
        ndecryptors: int,
    ) -> None:
        """
        Set the scenario parameters for FlamingoServer.

        Args:
            dimension (int): The dimension.
            valuesize (int): The value size.
            keysize (int): The key size.
            threshold (int): The threshold.
            nclients (int): The number of clients.
            neighborood_size (int): The neighborhood size.
            ndecryptors (int): The number of decryptors.
        """
        FlamingoServer.dimension = dimension
        FlamingoServer.value_size = valuesize
        FlamingoServer.nclients = nclients
        FlamingoServer.expanded_value_size = valuesize + ceil(log2(nclients))
        FlamingoServer.keysize = keysize
        FlamingoServer.key_length = ceil(keysize / 8)
        FlamingoServer.threshold = threshold
        FlamingoServer.prg = PRG(dimension, valuesize)
        FlamingoServer.ss_b_mask = SSS(PRG.security)
        FlamingoServer.prf = PRF(dimension, valuesize)
        FlamingoServer.teg = TElGamal(threshold, ndecryptors)
        FlamingoServer.neighborood_size = neighborood_size
        FlamingoServer.ndecryptors = ndecryptors
        FlamingoServer.decryptors = [i + 1 for i in range(ndecryptors)]

    def setup(self) -> Tuple[Any, Dict[int, Any]]:
        """
        Perform the setup for FlamingoServer.

        Returns:
            Tuple[Any, Dict[int, Any]]: Public key and key shares.
        """
        pub_key, key_shares = FlamingoServer.teg.setup()
        self.key_shares = key_shares
        return pub_key, key_shares

    def new_fl_step(self) -> None:
        """
        Advance to the next Flamingo step and reset internal state.
        """
        self.step += 1
        self.clients = []
        self.clients_on = []
        self.clients_off = []
        self.all_y = {}

    def setup_send_shares_commits_sk(
        self,
        all_shares_a: Dict[int, Dict[int, Any]],
        all_shares_b: Dict[int, Dict[int, Any]],
        all_commitments: List[Any],
    ) -> Tuple[Dict[int, Dict[int, Any]], Dict[int, Dict[int, Any]], List[Any]]:
        """
        Setup and send shares and commitments for secret key.

        Args:
            all_shares_a (Dict[int, Dict[int, Any]]): Dictionary of 'a' shares.
            all_shares_b (Dict[int, Dict[int, Any]]): Dictionary of 'b' shares.
            all_commitments (List[Any]): List of commitments.

        Returns:
            Tuple[Dict[int, Dict[int, Any]], Dict[int, Dict[int, Any]], List[Any]]: Dictionaries of shares 'a' and 'b',
            and a list of commitments.
        """
        self.all_shares_a = defaultdict(dict)
        self.all_shares_b = defaultdict(dict)
        self.all_commitments = all_commitments
        for user in all_shares_a.keys():
            for vuser in all_shares_a[user].keys():
                self.all_shares_a[vuser][user] = all_shares_a[user][vuser]
                self.all_shares_b[vuser][user] = all_shares_b[user][vuser]
        return self.all_shares_a, self.all_shares_b, self.all_commitments

    def setup_accept_or_complain_sk(
        self, complaints: Dict[int, List[int]]
    ) -> List[int]:
        """
        Accept or complain based on complaints towards secret key setup.

        Args:
            complaints (Dict[int, List[int]]): Dictionary mapping user identifiers to lists of complaints.

        Returns:
            List[int]: List of complaints.
        """
        self.complaints = list(complaints.values())
        self.complaints = [item for sublist in self.complaints for item in sublist]
        return self.complaints

    def setup_forward_shares(
        self,
        all_bcast_shares_a: Dict[int, List[Union[Dict[int, Any], Any]]],
        all_bcast_shares_b: Dict[int, List[Union[Dict[int, Any], Any]]],
    ) -> Tuple[List[Union[Dict[int, Any], Any]], List[Union[Dict[int, Any], Any]]]:
        """
        Forward shares for secret key setup and disqualify invalid ones.

        Args:
            all_bcast_shares_a (Dict[int, List[Union[Dict[int, Any], Any]]]): Dictionary of broadcast shares 'a'.
            all_bcast_shares_b (Dict[int, List[Union[Dict[int, Any], Any]]]): Dictionary of broadcast shares 'b'.

        Returns:
            Tuple[List[Union[Dict[int, Any], Any]], List[Union[Dict[int, Any], Any]]]: Lists of broadcast shares 'a' and 'b'.
        """
        all_bcast_shares_a = list(all_bcast_shares_a.values())
        self.all_bcast_shares_a = [
            item for sublist in all_bcast_shares_a for item in sublist
        ]
        all_bcast_shares_b = list(all_bcast_shares_b.values())
        self.all_bcast_shares_b = [
            item for sublist in all_bcast_shares_b for item in sublist
        ]
        return self.all_bcast_shares_a, self.all_bcast_shares_b

    def setup_broadcast_qual(
        self, all_qual: List[Union[Dict[int, Any], Any]]
    ) -> List[Union[Dict[int, Any], Any]]:
        """
        Broadcast qualification information.

        Args:
            all_qual (List[Union[Dict[int, Any], Any]]): List of qualification information.

        Returns:
            List[Union[Dict[int, Any], Any]]: List of qualification information.
        """
        self.all_qual = all_qual
        return self.all_qual

    def setup_send_commits_pk(self, all_commitments: List[Any]) -> List[Any]:
        """
        Send commitments for public key setup.

        Args:
            all_commitments (List[Any]): List of commitments.

        Returns:
            List[Any]: List of commitments.
        """
        self.all_commitments = all_commitments
        return self.all_commitments

    def advertise_keys(
        self, all_dh_pks: Dict[int, Any], all_dh_pkc: Dict[int, Any]
    ) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        """
        Advertise public keys.

        Args:
            all_dh_pks (Dict[int, Any]): All DH public keys.
            all_dh_pkc (Dict[int, Any]): All DH public keys for channels.

        Returns:
            Tuple[Dict[int, Any], Dict[int, Any]]: All DH public keys.
        """
        self.clients = list(all_dh_pkc.keys())
        assert all_dh_pkc.keys() == all_dh_pks.keys()
        assert (
            len(self.clients) >= FlamingoServer.threshold
        ), f"{len(self.clients)} < {FlamingoServer.threshold}"

        self.all_dh_pks = all_dh_pks

        return all_dh_pks, all_dh_pkc

    def report_share_keys(
        self,
        all_e_shares: Dict[int, Dict[int, Any]],
        all_e_messages: Dict[Tuple[int, int], Any],
    ) -> None:
        """
        Report shared keys.

        Args:
            all_e_shares (Dict[int, Dict[int, Any]]): All encrypted shares.
            all_e_messages (Dict[Tuple[int, int], Any]): All encrypted messages.
        """
        assert len(self.clients) >= FlamingoServer.threshold

        self.all_e_shares = all_e_shares

        for key, value in all_e_messages.items():
            for item in value:
                self.all_e_messages[item] = all_e_messages[key][item]

    def cross_check(
        self, all_y: Dict[int, List[int]]
    ) -> Tuple[Dict[int, Dict[int, Any]], Dict[Tuple[int, int], Any]]:
        """
        Cross-check and send requests to decrypt pairwise seeds and encrypted shares.

        Args:
            all_y (Dict[int, List[int]]): Encrypted shares.

        Returns:
            Tuple[Dict[int, Dict[int, Any]], Dict[Tuple[int, int], Any]]: Encrypted shares
            and encrypted messages.
        """
        self.clients_on = list(all_y.keys())
        self.clients_off = list(set(self.clients) - set(self.clients_on))

        assert len(self.clients_on) >= FlamingoServer.threshold
        assert set(self.clients_on).issubset(set(self.clients))

        self.all_y = all_y
        all_e_shares = defaultdict(dict)

        for user in self.clients_on:
            for vuser in FlamingoServer.decryptors:
                if user == vuser:
                    continue
                all_e_shares[vuser][user] = self.all_e_shares[user][vuser]

        all_e_messages = defaultdict(dict)
        self.target_pairwise = []
        for c_off in self.clients_off:
            c_off_neighbors = find_neighbors(
                self.step,
                FlamingoServer.nclients,
                c_off,
                FlamingoServer.neighborood_size,
            )
            for user in c_off_neighbors:
                if user in self.clients_on:
                    self.target_pairwise.append((c_off, user))
                    all_e_messages[(c_off, user)] = self.all_e_messages[(c_off, user)]

        return all_e_shares, all_e_messages

    def reconstruction(
        self,
        all_sk_shares: Dict[int, Dict[Tuple[int, int], List[int]]],
        all_b_shares: Dict[int, Dict[int, List[int]]],
    ) -> List[int]:
        """
        Reconstruct pairwise seeds and blinding mask.

        Args:
            all_sk_shares (Dict[int, Dict[Tuple[int, int], List[int]]]): All secret key shares.
            all_b_shares (Dict[int, Dict[int, List[int]]]): All blinding mask shares.

        Returns:
            List[int]: Reconstructed result.
        """
        b_shares = defaultdict(list)
        for user in all_b_shares:
            for off_user in all_b_shares[user]:
                b_shares[off_user].append(all_b_shares[user][off_user])

        lag_coeffs = []
        b_mask_vector_result = [0] * FlamingoServer.dimension
        for user in b_shares:
            assert len(b_shares[user]) >= FlamingoServer.threshold
            if not lag_coeffs:
                lag_coeffs = FlamingoServer.ss_b_mask.lagrange(
                    b_shares[user][: FlamingoServer.threshold]
                )
            b_mask = FlamingoServer.ss_b_mask.reconstruct(
                b_shares[user][: FlamingoServer.threshold],
                FlamingoServer.threshold,
                lag_coeffs,
            )
            b_mask_vector = FlamingoServer.prg.eval_vector(b_mask)
            b_mask_vector_result = add_vectors(
                b_mask_vector_result,
                b_mask_vector,
                2**FlamingoServer.expanded_value_size,
            )
            del b_mask_vector
            gc.collect()

        # TODO not efficient to compute the lagrange coefficients twice, but the one for the blinding mask is over a Field_128 whereas for the TElgamal it is over a Field_256
        lag_coeffs = FlamingoServer.teg.ss.lagrange(
            self.key_shares[: FlamingoServer.threshold]
        )

        partial_decryptions = defaultdict(list)
        for user in all_sk_shares:
            for pair in self.target_pairwise:
                partial_decryptions[pair].append(all_sk_shares[user][pair])

        skey = [0] * FlamingoServer.dimension
        for key, value in partial_decryptions.items():
            seed_group = FlamingoServer.teg.decrypt(
                value[: FlamingoServer.threshold], self.all_e_messages[key], lag_coeffs
            )
            pairwise_seed = FlamingoServer.prf.to_hash(seed_group)
            sv = FlamingoServer.prf.eval_vector(pairwise_seed)
            user, vuser = key
            if vuser > user:
                skey = sub_vectors(skey, sv, 2**FlamingoServer.expanded_value_size)
            else:
                skey = add_vectors(skey, sv, 2**FlamingoServer.expanded_value_size)

        result = [0] * FlamingoServer.dimension
        for user in self.all_y:
            result = add_vectors(
                result, self.all_y[user], 2**FlamingoServer.expanded_value_size
            )

        result = sub_vectors(
            result, b_mask_vector_result, 2**FlamingoServer.expanded_value_size
        )
        del b_mask_vector_result
        gc.collect()
        result = add_vectors(result, skey, 2**FlamingoServer.expanded_value_size)

        return result
