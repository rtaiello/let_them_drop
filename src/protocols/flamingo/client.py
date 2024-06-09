"""
Implementation of Flamingo protocol https://eprint.iacr.org/2023/486.pdf.
"""
import gc
from collections import defaultdict
from math import ceil, log2
from os import urandom as rng
from random import randint
from typing import Any, Dict, List, Tuple

import gmpy2

from ...buildingblocks import KAS, PRF, PRG, SSS
from ...buildingblocks import EncryptionKey as AESKEY
from ...buildingblocks import Share, TElGamal, add_vectors, sub_vectors
from ..flamingo.utils import commit, find_neighbors, verify_commit


class FlamingoClient(object):
    dimension: int = 1000
    value_size: int = 16
    nclients: int = 10
    expanded_value_size: int = value_size + ceil(log2(nclients))
    keysize: int = 256
    threshold: int = ceil(2 * nclients / 3)
    clients: List[int] = [i + 1 for i in range(nclients)]
    ss_b_mask: SSS = SSS(keysize)

    def __init__(self, user: int, key_share: Any, pub_key: Any) -> None:
        """
        Initialize a FlamingoClient instance.

        Args:
            user (int): The user identifier.
            key_share (Any): The key share.
            pub_key (Any): The public key.
        """
        super().__init__()
        self.user = user
        self.step = 1
        self.all_public_keys: Dict[int, Any] = {}
        self.channel_keys: Dict[int, Any] = {}
        self.clients: List[int] = []
        self.clients_on: List[int] = []
        self.clients_off: List[int] = []
        self.b_mask_shares: Dict[int, Share] = {}
        self.x: List[int] = []
        self.ka_channel = KAS()
        self.rcv_shares_a = {}
        self.rcv_shares_b = {}
        self.ka_sk = KAS()
        self.b_mask: int = 0
        self.secrets: Dict[int, bytes] = {}
        self.seeds: Dict[int, bytes] = {}
        self.key_share = key_share
        self.public_key = pub_key
        self.decryptors: List[int] = [i + 1 for i in range(60)]

    @staticmethod
    def set_scenario(
        dimension: int,
        value_size: int,
        keysize: int,
        threshold: int,
        nclients: int,
        neighborood_size: int,
        ndecryptors: int,
    ) -> None:
        """
        Set the scenario parameters for FlamingoClient.

        Args:
            dimension (int): The dimension.
            value_size (int): The value size.
            keysize (int): The key size.
            threshold (int): The threshold.
            nclients (int): The number of clients.
            neighborood_size (int): The neighborhood size.
            ndecryptors (int): The number of decryptors.
        """
        FlamingoClient.dimension = dimension
        FlamingoClient.value_size = value_size
        FlamingoClient.nclients = nclients
        FlamingoClient.clients = [i + 1 for i in range(nclients)]
        FlamingoClient.expanded_value_size = value_size + ceil(log2(nclients))
        FlamingoClient.keysize = keysize
        FlamingoClient.key_length = ceil(keysize / 8)
        FlamingoClient.threshold = threshold
        FlamingoClient.prg = PRG(dimension, value_size)
        FlamingoClient.prf = PRF(dimension, value_size)
        FlamingoClient.ss_b_mask = SSS(PRG.security)
        FlamingoClient.teg = TElGamal(threshold, ndecryptors)
        FlamingoClient.neighborood_size = neighborood_size
        FlamingoClient.ndecryptors = ndecryptors
        FlamingoClient.decryptors = [i + 1 for i in range(ndecryptors)]
        rand_tmp = randint(1, FlamingoClient.teg.curve_params.order)
        FlamingoClient.g = FlamingoClient.teg.curve_params.p
        FlamingoClient.h = FlamingoClient.g * rand_tmp

    def new_fl_step(self) -> None:
        """
        Advance to the next Flamingo step and reset internal state.
        """
        self.step += 1
        self.clients = []
        self.clients_on = []
        self.clients_off = []
        self.b_mask_shares = {}
        self.keys_encrypted = {}
        self.b_mask = 0
        self.secrets = {}
        self.seeds = {}
        self.neighbors = find_neighbors(
            self.step,
            FlamingoClient.nclients,
            self.user,
            FlamingoClient.neighborood_size,
        )

    def advertise_keys(self) -> Tuple[int, Any, Any]:
        """
        Advertise public keys.

        Returns:
            Tuple[int, Any, Any]: User identifier, secret key, and channel key.
        """
        self.ka_sk.generate()
        self.ka_channel.generate()
        self.clients.append(self.user)
        return self.user, self.ka_sk.pk, self.ka_channel.pk

    def setup_send_shares_commits_sk(
        self,
    ) -> Tuple[int, Dict[int, bytes], Dict[int, bytes], List[Any]]:
        """
        Setup and send shares and commitments for secret key.

        Args:
        Returns:
            Tuple[int, Dict[int, bytes], Dict[int, bytes], List[Any]]: User identifier, encrypted shares of 'a' and 'b',
            and list of commitments.
        """
        a = randint(1, FlamingoClient.teg.curve_params.order)
        b = randint(1, FlamingoClient.teg.curve_params.order)

        self.shares_a, coeffs_a = FlamingoClient.teg.ss.share(
            a, FlamingoClient.threshold, FlamingoClient.nclients, get_coeffs=True
        )
        self.shares_b, coeffs_b = FlamingoClient.teg.ss.share(
            b, FlamingoClient.threshold, FlamingoClient.nclients, get_coeffs=True
        )
        my_commitments = []
        for j in range(0, FlamingoClient.threshold):
            commitement_a = commit(coeffs_a[j], FlamingoClient.g)
            commitement_b = commit(coeffs_b[j], FlamingoClient.h)
            my_commitments.append(commitement_a + commitement_b)
        self.coeffs_a = coeffs_a  # Store coefficients for later use
        e_shares_a = {}
        e_shares_b = {}
        for share_a, share_b in zip(self.shares_a, self.shares_b):
            vuser = share_a.idx
            if self.user == vuser:
                self.rcv_shares_a[self.user] = share_a
                self.rcv_shares_b[self.user] = share_b
                continue
            key = AESKEY(self.channel_keys[vuser])
            message_a = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + gmpy2.to_binary(share_a.value._value)
            )
            message_b = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + gmpy2.to_binary(share_b.value._value)
            )
            e_a = key.encrypt(message_a)
            e_shares_a[vuser] = e_a
            e_b = key.encrypt(message_b)
            e_shares_b[vuser] = e_b
        return self.user, e_shares_a, e_shares_b, my_commitments

    def setup_receive_shares_commits_sk(
        self,
        e_shares_a: Dict[int, bytes],
        e_shares_b: Dict[int, bytes],
        commitments: List[Any],
    ) -> List[int]:
        """
        Setup for receiving shares and commitments for secret key and verify them.

        Args:
            e_shares_a (Dict[int, bytes]): Dictionary mapping user identifiers to encrypted shares of 'a'.
            e_shares_b (Dict[int, bytes]): Dictionary mapping user identifiers to encrypted shares of 'b'.
            commitments (List[Any]): List of commitments.

        Returns:
            List[int]: List of user identifiers towards whom complaints exist.
        """
        complaints_towards = []

        assert len(e_shares_a) >= FlamingoClient.threshold
        for vuser in e_shares_a:
            key = AESKEY(self.channel_keys[vuser])
            message_a = key.decrypt(e_shares_a[vuser])
            u = int.from_bytes(message_a[:2], "big")
            v = int.from_bytes(message_a[2:4], "big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            share_a = gmpy2.from_binary(message_a[4:])
            field = FlamingoClient.teg.ss.Field
            self.rcv_shares_a[vuser] = Share(self.user, field(share_a))
            message_b = key.decrypt(e_shares_b[vuser])
            u = int.from_bytes(message_b[:2], "big")
            v = int.from_bytes(message_b[2:4], "big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            share_b = gmpy2.from_binary(message_b[4:])
            self.rcv_shares_b[vuser] = Share(self.user, field(share_b))

        for id in FlamingoClient.clients:
            if (
                verify_commit(
                    self.rcv_shares_a[id],
                    self.rcv_shares_b[id],
                    commitments[id],
                    FlamingoClient.g,
                    FlamingoClient.h,
                    self.user,
                )
                == False
            ):
                complaints_towards.append(id)
        return complaints_towards

    def setup_accept_or_complain_sk(
        self, complaints_towards: List[int]
    ) -> Tuple[Dict[int, Share], Dict[int, Share]]:
        """
        Accept or complain based on complaints towards secret key setup.

        Args:
            complaints_towards (List[int]): List of user identifiers towards whom complaints exist.

        Returns:
            Tuple[Dict[int, Share], Dict[int, Share]]: Dictionaries mapping user identifiers to shares 'a' and 'b'.
        """
        bcast_shares_a = {}
        b_cast_shares_b = {}
        complaint_list = complaints_towards
        for complain_party in complaint_list:
            bcast_shares_a[complain_party] = self.shares_a[complain_party]
            b_cast_shares_b[complain_party] = self.shares_b[complain_party]
        return bcast_shares_a, b_cast_shares_b

    def setup_forward_shares(
        self,
        bcast_shares_a: Dict[int, List[Share]],
        bcast_shares_b: Dict[int, List[Share]],
    ) -> List[int]:
        """
        Forward shares for secret key setup and disqualify invalid ones.

        Args:
            bcast_shares_a (Dict[int, List[Share]]): Dictionary mapping user identifiers to shares 'a'.
            bcast_shares_b (Dict[int, List[Share]]): Dictionary mapping user identifiers to shares 'b'.

        Returns:
            List[int]: List of user identifiers with valid shares.
        """
        rcv_bcast_shares_a = bcast_shares_a
        rcv_bcast_shares_b = bcast_shares_b

        disqual = []
        for id in rcv_bcast_shares_a:
            if len(rcv_bcast_shares_a[id]) == 0:
                continue
            else:
                if (
                    verify_commit(
                        rcv_bcast_shares_a[id],
                        rcv_bcast_shares_b[id],
                        FlamingoClient.g,
                        FlamingoClient.h,
                        id,
                    )
                    == False
                ):
                    disqual.append(id)

        qual = set(self.rcv_shares_a.keys()) - set(disqual)

        return qual

    def setup_create_sk(self, quals_dict: Dict[int, List[int]]) -> int:
        """
        Create secret key based on qualified users.

        Args:
            quals_dict (Dict[int, List[int]]): Dictionary mapping user identifiers to a list of qualified users.

        Returns:
            int: Secret key share.
        """
        qual_list = list(quals_dict.values())
        agreed_qual = max(qual_list, key=qual_list.count)
        sk_share = 0
        for id in agreed_qual:
            sk_share = (
                sk_share + self.rcv_shares_a[id].value._value
            ) % FlamingoClient.teg.curve_params.order
        return sk_share

    def setup_send_commits_pk(self) -> List[Any]:
        """
        Send commitments for public key setup.

        Returns:
            List[Any]: List of commitments.
        """
        my_commitments = []
        for j in range(0, FlamingoClient.threshold):
            commitement_a = commit(self.coeffs_a[j], FlamingoClient.g)
            my_commitments.append(commitement_a)
        return my_commitments

    def setup_receive_commits_pk(self, commitments: List[Any]) -> List[int]:
        """
        Receive commitments for public key setup and verify them.

        Args:
            commitments (List[Any]): List of commitments.

        Returns:
            List[int]: List of user identifiers towards whom complaints exist.
        """
        complaints_towards = []
        self.commitments_a = commitments
        for id in FlamingoClient.clients:
            if (
                verify_commit(
                    self.rcv_shares_a[id],
                    None,
                    commitments[id],
                    FlamingoClient.g,
                    None,
                    self.user,
                )
                == False
            ):
                complaints_towards.append(id)
        return complaints_towards

    def setup_create_pk(self) -> Any:
        """
        Create public key based on received commitments.

        Returns:
            Any: Public key.
        """
        pk = self.commitments_a[1][FlamingoClient.threshold - 1]
        for i in range(2, len(self.commitments_a) + 1):
            pk = pk + self.commitments_a[i][FlamingoClient.threshold - 1]
        return pk

    def report_pairwise_secrets(
        self, all_dh_pks: Dict[int, Any], all_dh_pkc: Dict[int, Any]
    ) -> None:
        """
        Report pairwise secrets.

        Args:
            all_dh_pks (Dict[int, Any]): All DH public keys.
            all_dh_pkc (Dict[int, Any]): All DH public keys for channels.
        """
        for vuser in self.neighbors:
            if vuser == self.user:
                continue
            secret = self.ka_sk.agree(all_dh_pks[vuser])
            # convert to bytes
            self.secrets[vuser] = int(secret).to_bytes(FlamingoClient.key_length, "big")
        for vuser in all_dh_pkc:
            if vuser == self.user:
                continue
            self.channel_keys[vuser] = self.ka_channel.agree(all_dh_pkc[vuser])

    def report_share_keys(
        self,
    ) -> Tuple[int, Dict[int, Any], Dict[Tuple[int, int], Any]]:
        """
        Report shared keys.

        Returns:
            Tuple[int, Dict[int, Any], Dict[Tuple[int, int], Any]]: User identifier,
            encrypted shares, and encrypted messages.
        """
        neighbor_pairwise_mask_seed = {}
        e_messages = defaultdict(dict)
        for vuser in self.neighbors:
            if vuser == self.user:
                continue

            neighbor_pairwise_mask_seed[vuser] = FlamingoClient.prf.eval_key(
                key=self.secrets[vuser], round=self.step
            )
            self.seeds[vuser] = FlamingoClient.prf.to_hash(
                neighbor_pairwise_mask_seed[vuser]
            )
            message = neighbor_pairwise_mask_seed[vuser]

            e_messages[(self.user, vuser)] = FlamingoClient.teg.encrypt(
                message, self.public_key
            )

        self.b_mask = randint(0, 2**FlamingoClient.prg.security - 1)
        b_shares = FlamingoClient.ss_b_mask.share(
            self.b_mask, FlamingoClient.threshold, FlamingoClient.ndecryptors
        )
        e_shares = {}
        for bshare in b_shares:
            vuser = bshare.idx
            if self.user == vuser:
                self.b_mask_shares[self.user] = bshare
                continue
            key = AESKEY(self.channel_keys[vuser])
            sharelen = len(gmpy2.to_binary(bshare.value._value))
            message = (
                self.user.to_bytes(2, "big")
                + vuser.to_bytes(2, "big")
                + sharelen.to_bytes(2, "big")
                + gmpy2.to_binary(bshare.value._value)
            )
            e = key.encrypt(message)
            e_shares[vuser] = e

        return self.user, e_shares, e_messages

    def report_masked_input(self) -> Tuple[int, List[int]]:
        """
        Report masked input.

        Returns:
            Tuple[int, List[int]]: User identifier and masked input.
        """
        key = [0] * FlamingoClient.dimension
        for vuser in self.neighbors:
            if vuser == self.user:
                continue
            sv = FlamingoClient.prf.eval_vector(seed=self.seeds[vuser])
            if vuser > self.user:
                key = sub_vectors(key, sv, 2**FlamingoClient.expanded_value_size)
            else:
                key = add_vectors(key, sv, 2**FlamingoClient.expanded_value_size)
            del sv
        b_mask_vector = FlamingoClient.prg.eval_vector(self.b_mask)
        x_masked = add_vectors(
            key, b_mask_vector, 2**FlamingoClient.expanded_value_size
        )
        del key
        x = [1 for _ in range(FlamingoClient.dimension)]
        y = add_vectors(x, x_masked, 2**FlamingoClient.expanded_value_size)
        del x
        del x_masked
        gc.collect()
        return self.user, y

    def reconstruction(
        self, e_shares: Dict[int, Any], e_messages: Dict[Tuple[int, int], Any]
    ) -> Tuple[int, Dict[Tuple[int, int], Any], Dict[int, Share]]:
        """
        Perform reconstruction.

        Args:
            e_shares (Dict[int, Any]): Encrypted shares.
            e_messages (Dict[Tuple[int, int], Any]): Encrypted messages.

        Returns:
            Tuple[int, Dict[Tuple[int, int], Any], Dict[int, Share]]: User identifier,
            partial decryptions, and masked input shares.
        """
        assert len(e_shares) + 1 >= FlamingoClient.threshold

        for vuser in e_shares:
            key = AESKEY(self.channel_keys[vuser])
            message = key.decrypt(e_shares[vuser])
            u = int.from_bytes(message[:2], "big")
            v = int.from_bytes(message[2:4], "big")
            sharelen = int.from_bytes(message[4:6], "big")
            assert v == self.user and u == vuser, "invalid encrypted message"
            bshare = gmpy2.from_binary(message[6 : sharelen + 6])
            self.b_mask_shares[vuser] = Share(
                self.user, FlamingoClient.ss_b_mask.Field(bshare)
            )

        partial_decryptions = defaultdict(dict)
        for key, value in e_messages.items():
            partial_decryptions[key] = FlamingoClient.teg.share_decrypt(
                value, self.key_share
            )
        return self.user, partial_decryptions, self.b_mask_shares
