import random
from math import factorial
from typing import Dict, List, Optional, Union

from gmpy2 import mpz

from ...full_domain_hash import FDH
from ...ss.shoup_ss import ShoupShare, ShoupSSS
from ...utils import get_two_safe_primes, powmod
from ..jl import DEFAULT_KEY_SIZE, JLS
from ..jl_utils import EncryptedNumberJL, PublicParamJL, ServerKeyJL, UserKeyJL
from ..vector_encoding import VES


class TD_TJLS(JLS):
    def __init__(
        self,
        threshold: int,
        nusers: int,
        ve: Optional[VES] = None,
    ) -> None:
        """
        Initializes a TD_TJLS instance.

        Args:
            threshold (int): The threshold for sharing and reconstruction.
            nusers (int): The number of users in the system.
            ve (VES, optional): An instance of VES for vector encoding. Defaults to None.
        """
        super().__init__(nusers, ve)
        self.threshold = threshold
        self.delta = factorial(self.nusers)
        self.shoup_ss: Optional[ShoupSSS] = None

    def setup(self, lmbda: int = DEFAULT_KEY_SIZE) -> PublicParamJL:
        """
        Performs the system setup to generate public parameters, server key, and user keys.

        Args:
            lmbda (int, optional): The security parameter. Defaults to DEFAULT_KEY_SIZE.

        Returns:
            PublicParamJL: Public parameters.
        """
        self.keysize: int = lmbda
        p, q = get_two_safe_primes(lmbda)
        pp = (p - 1) // 2
        qq = (q - 1) // 2
        p = 2 * pp + 1
        q = 2 * qq + 1
        n = p * q
        self.phi_n = pp * qq

        fdh: FDH = FDH(self.keysize, n * n)

        public_param: PublicParamJL = PublicParamJL(n, lmbda // 2, fdh.H)

        seed: random.Random = random.SystemRandom()
        n_len: int = n.bit_length()
        s0: mpz = mpz(0)
        users: Dict[int, UserKeyJL] = {}
        for i in range(self.nusers):
            s: mpz = mpz(seed.getrandbits(2 * n_len))
            users[i] = UserKeyJL(public_param, s)
            s0 += s
        s0 = -s0
        server: ServerKeyJL = ServerKeyJL(public_param, s0)

        self.shoup_ss = ShoupSSS(n)

        return public_param, server, users

    def sk_share(self, sk_u: UserKeyJL) -> List[ShoupShare]:
        """
        Shares a user's key using Shoup's secret sharing scheme.

        Args:
            sk_u (UserKeyJL): User's key to share.

        Returns:
            List[ShoupShare]: List of shared key shares.
        """
        return self.shoup_ss.share(sk_u.s, self.threshold, self.nusers, self.phi_n)

    def share_protect(
        self, pp: PublicParamJL, list_sk_v_ushare: List[ShoupShare], tau: int
    ) -> Union[ShoupShare, List[ShoupShare]]:
        """
        Protects a list of shared keys.

        Args:
            pp (PublicParamJL): Public parameters.
            list_sk_v_ushare (List[ShoupShare]): List of shared keys to protect.
            tau (int): Time parameter.

        Returns:
            Union[ShoupShare, List[ShoupShare]]: The protected shared keys.
        """
        sum_share = list_sk_v_ushare[0]
        idx = sum_share.idx
        for share in list_sk_v_ushare[1:]:
            sum_share += share
        key_share = UserKeyJL(pp, sum_share.value)
        if self.ve is not None:
            yzero_u_tau_shares = self.protect(
                pp, key_share, tau, [0] * self.ve.vectorsize
            )
            r = []
            ShoupShare.bits = yzero_u_tau_shares[0].ciphertext.bit_length()
            for yzero_share in yzero_u_tau_shares:
                # TODO FIX BUG HERE
                r.append(ShoupShare(idx, yzero_share))
            return r
        else:
            yzero_u_tau_shares = self.protect(pp, key_share, tau, 0)
            ShoupShare.bits = yzero_u_tau_shares.ciphertext.bit_length()
            return ShoupShare(idx, yzero_u_tau_shares)

    def share_combine(
        self,
        pp: PublicParamJL,
        yzero_tau_shares: List[Union[EncryptedNumberJL, List[EncryptedNumberJL]]],
        threshold: int,
    ) -> Union[EncryptedNumberJL, List[EncryptedNumberJL]]:
        """
        Combines and reconstructs shared keys.

        Args:
            pp (PublicParamJL): Public parameters.
            yzero_tau_shares (List[Union[EncryptedNumberJL, List[EncryptedNumberJL]]]): List of shared keys to combine and reconstruct.
            threshold (int): The threshold for reconstruction.

        Returns:
            Union[EncryptedNumberJL, List[EncryptedNumberJL]]: The combined and reconstructed shared keys.
        """

        def _reconstruct(
            shares: List[EncryptedNumberJL],
            threshold: int,
            delta: int,
            lag_coeffs: List[mpz] = None,
        ) -> EncryptedNumberJL:
            assert len(shares) >= threshold, "not enough shares, cannot reconstruct!"
            raw_shares = []
            for x in shares:
                idx = x.idx
                value = x.value
                if any(y[0] == idx for y in raw_shares):
                    raise ValueError("Duplicate share")
                raw_shares.append((idx, value))
            k = len(shares)
            product = 1
            for j in range(k):
                x_j, y_j = raw_shares[j]
                r = powmod(y_j.ciphertext, lag_coeffs[x_j], pp.n_squared)
                product = (product * r) % pp.n_squared
            return EncryptedNumberJL(pp, product)

        if isinstance(yzero_tau_shares[0], list):
            l = len(yzero_tau_shares[0])
            for vshare in yzero_tau_shares:
                assert l == len(
                    vshare
                ), "shares of the vector do not have the same size"

            vector_recon = []
            lag_coeffs = None
            for counter in range(l):
                element_shares = []
                for vshare in yzero_tau_shares:
                    element_shares.append(vshare[counter])
                if lag_coeffs is None:
                    lag_coeffs = self.shoup_ss.lagrange(element_shares, self.delta)
                vector_recon.append(
                    _reconstruct(element_shares, threshold, self.delta, lag_coeffs)
                )
            return vector_recon
        else:
            lag_coeffs = self.shoup_ss.lagrange(yzero_tau_shares, self.delta)
            return _reconstruct(yzero_tau_shares, threshold, self.delta, lag_coeffs)

    def agg(
        self,
        pp: PublicParamJL,
        sk_0: ServerKeyJL,
        tau: int,
        list_y_u_tau: List[Union[List[EncryptedNumberJL], EncryptedNumberJL]],
        yzero_tau: Optional[List[EncryptedNumberJL]] = None,
    ) -> int:
        """
        Aggregates the protected inputs and computes the sum.

        Args:
            pp (PublicParamJL): Public parameters.
            sk_0 (ServerKeyJL): Server key.
            tau (int): Time parameter.
            list_y_u_tau (List[Union[List[EncryptedNumberJL], EncryptedNumberJL]]): List of protected inputs.
            yzero_tau (List[EncryptedNumberJL], optional): Optional list of shared keys for aggregation. Defaults to None.

        Returns:
            int: The sum of the protected inputs.
        """
        assert isinstance(sk_0, ServerKeyJL), "bad server key"
        assert sk_0.pp == pp, "bad server key"
        assert isinstance(list_y_u_tau, list), "list_y_u_tau should be a list"
        assert (
            len(list_y_u_tau) > 0
        ), "list_y_u_tau should contain at least one protected input"

        if isinstance(list_y_u_tau[0], list):
            if yzero_tau:
                assert len(list_y_u_tau[0]) == len(yzero_tau), "bad vector length"
            for y_u_tau in list_y_u_tau:
                assert len(y_u_tau) == len(list_y_u_tau[0]), "bad vector length"

            y_tau = []
            delta = 1
            for i in range(len(list_y_u_tau[0])):
                y_tau_i = list_y_u_tau[0][i]
                for y_u_tau in list_y_u_tau[1:]:
                    y_tau_i += y_u_tau[i]
                if yzero_tau:
                    y_tau_i = EncryptedNumberJL(
                        pp, powmod(y_tau_i.ciphertext, self.delta, sk_0.pp.n_squared)
                    )
                    y_tau_i += yzero_tau[i]
                    delta = self.delta
                y_tau.append(y_tau_i)
            d = sk_0.decrypt(y_tau, tau, delta, True)
            sum_x_u_tau = self.ve.decode(d)

        else:
            assert isinstance(list_y_u_tau[0], EncryptedNumberJL), "bad ciphertext"
            y_tau = list_y_u_tau[0]
            delta = 1
            for y_u_tau in list_y_u_tau[1:]:
                y_tau += y_u_tau
            if yzero_tau:
                y_tau = EncryptedNumberJL(
                    pp, powmod(y_tau.ciphertext, self.delta, sk_0.pp.n_squared)
                )
                y_tau += yzero_tau
                delta = self.delta
            sum_x_u_tau = sk_0.decrypt(y_tau, tau, delta, True)
        return sum_x_u_tau
