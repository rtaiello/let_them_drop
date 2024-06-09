from math import factorial
from typing import List, Optional, Tuple, Union

from gmpy2 import mpz, powmod

from ...ss.integer_ss import ISSS, IShare
from ..jl import DEFAULT_KEY_SIZE, JLS
from ..jl_utils import EncryptedNumberJL, PublicParamJL, ServerKeyJL, UserKeyJL
from ..vector_encoding import VES


class TJLS(JLS):
    def __init__(
        self,
        threshold: int,
        nusers: int,
        ve: Optional[VES] = None,
        sigma: int = 128,
    ) -> None:
        """
        Initializes a TJLS instance.

        Args:
            threshold (int): The threshold for sharing and reconstruction.
            nusers (int): The number of users in the system.
            ve (VES, optional): An instance of VES for vector encoding. Defaults to None.
            sigma (int, optional): The sigma parameter. Defaults to 128.
        """
        assert threshold <= nusers, "threshold must be less than or equal to nusers"
        super().__init__(nusers, ve)
        self.threshold = threshold
        self.delta = factorial(self.nusers)
        self.sigma = sigma
        self.iss: Optional[ISSS] = None

    def setup(
        self, lmbda: int = DEFAULT_KEY_SIZE
    ) -> Tuple[PublicParamJL, ServerKeyJL, List[UserKeyJL]]:
        """
        Performs the system setup to generate public parameters, server key, and user keys.

        Args:
            lmbda (int, optional): The security parameter. Defaults to DEFAULT_KEY_SIZE.

        Returns:
            Tuple[PublicParamJL, ServerKeyJL, List[UserKeyJL]]: A tuple containing public parameters,
            server key, and a list of user keys.
        """
        public_param, server, users = super().setup(lmbda)
        self.iss = ISSS(self.keysize, self.sigma)
        return public_param, server, users

    def sk_share(self, sk_u: UserKeyJL) -> List[IShare]:
        """
        Shares a user's key.

        Args:
            sk_u (UserKeyJL): User's key to share.

        Returns:
            List[IShare]: List of shared key shares.
        """
        return self.iss.share(sk_u.s, self.threshold, self.nusers)

    def share_protect(
        self, pp: PublicParamJL, list_sk_v_u_shares: List[IShare], tau: int
    ) -> Union[IShare, List[IShare]]:
        """
        Protects a zero value.

        Args:
            pp (PublicParamJL): Public parameters.
            list_sk_v_u_shares (List[IShare]): List of shared keys used to protect.
            tau (int): Time parameter.

        Returns:
            Union[IShare, List[IShare]]: The protected zero value.
        """
        sum_share = list_sk_v_u_shares[0]
        idx = sum_share.idx
        for share in list_sk_v_u_shares[1:]:
            sum_share += share
        key_share = UserKeyJL(pp, sum_share.value)
        if self.ve is not None:
            yzero_u_tau_shares = self.protect(
                pp, key_share, tau, [0] * self.ve.vectorsize
            )
            r = []
            IShare.bits = yzero_u_tau_shares[0].ciphertext.bit_length()
            for yzero_share in yzero_u_tau_shares:
                r.append(IShare(idx, yzero_share))
            return r
        else:
            yzero_u_tau_shares = self.protect(pp, key_share, tau, 0)
            return IShare(idx, yzero_u_tau_shares)

    def share_combine(
        self,
        pp: PublicParamJL,
        yzero_tau_shares: List[Union[EncryptedNumberJL, List[EncryptedNumberJL]]],
        threshold: int,
        lag_coeffs: List[mpz] = None,
    ) -> Union[EncryptedNumberJL, List[EncryptedNumberJL]]:
        """
        Combines and reconstructs the zero value.

        Args:
            pp (PublicParamJL): Public parameters.
            yzero_tau_shares (List[Union[EncryptedNumberJL, List[EncryptedNumberJL]]]): List of zero values to combine and reconstruct.
            threshold (int): The threshold for reconstruction.
            lag_coeffs (List[mpz], optional): Lagrange coefficients. Defaults to None.

        Returns:
            Union[EncryptedNumberJL, List[EncryptedNumberJL]]: The combined and reconstructed zero value.
        """
        assert len(yzero_tau_shares) > 0, "empty list of shares to reconstruct from"

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
            if not lag_coeffs:
                lag_coeffs = self.iss.lagrange(shares, delta)
            for j in range(k):
                x_j, y_j = raw_shares[j]
                r = powmod(y_j.ciphertext, lag_coeffs[x_j], pp.n_squared)
                product = (product * r) % pp.n_squared
            return EncryptedNumberJL(pp, product)

        if isinstance(yzero_tau_shares[0], list):
            l = len(yzero_tau_shares[0])
            for vector_share in yzero_tau_shares:
                assert l == len(
                    vector_share
                ), "shares of the vector do not have the same size"

            vector_recon = []
            lag_coeffs = []
            for counter in range(l):
                element_shares = []
                for vector_share in yzero_tau_shares:
                    element_shares.append(vector_share[counter])
                if not lag_coeffs:
                    lag_coeffs = self.iss.lagrange(element_shares, self.delta)
                vector_recon.append(
                    _reconstruct(element_shares, threshold, self.delta, lag_coeffs)
                )
            return vector_recon
        else:
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
        Aggregates protected inputs and computes the sum.

        Args:
            pp (PublicParamJL): Public parameters.
            sk_0 (ServerKeyJL): Server key.
            tau (int): Time parameter.
            list_y_u_tau (List[Union[List[EncryptedNumberJL], EncryptedNumberJL]]): List of protected inputs.
            yzero_tau (List[EncryptedNumberJL], optional): Optional list of shared keys for aggregation. Defaults to None.

        Returns:
            int: The sum of the protected inputs.
        """
        if isinstance(list_y_u_tau[0], list):
            y_tau = []
            delta = 1
            for i in range(len(list_y_u_tau[0])):
                y_tau_i = list_y_u_tau[0][i]
                for y_u_tau in list_y_u_tau[1:]:
                    y_tau_i += y_u_tau[i]
                if yzero_tau:
                    y_tau_i = EncryptedNumberJL(
                        pp,
                        powmod(y_tau_i.ciphertext, self.delta**2, sk_0.pp.n_squared),
                    )
                    y_tau_i += yzero_tau[i]
                    delta = self.delta
                y_tau.append(y_tau_i)

            d = sk_0.decrypt(y_tau, tau, delta)
            sum_x_u_tau = self.ve.decode(d)

        else:
            assert isinstance(list_y_u_tau[0], EncryptedNumberJL), "bad ciphertext"
            y_tau = list_y_u_tau[0]
            delta = 1
            for y_u_tau in list_y_u_tau[1:]:
                y_tau += y_u_tau
            if yzero_tau:
                y_tau = EncryptedNumberJL(
                    pp, powmod(y_tau.ciphertext, self.delta**2, sk_0.pp.n_squared)
                )
                y_tau += yzero_tau
                delta = self.delta
            sum_x_u_tau = sk_0.decrypt(y_tau, tau, delta)

        return sum_x_u_tau
