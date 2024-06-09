"""Code Implemented by:
https://github.com/MohamadMansouri/fault-tolerant-secure-agg/tree/main
"""
import random
from typing import Dict, List, Optional, Tuple, Union

from gmpy2 import mpz

from ..full_domain_hash import FDH
from ..utils import getprimeover
from .jl_utils import EncryptedNumberJL, PublicParamJL, ServerKeyJL, UserKeyJL
from .vector_encoding import VES

DEFAULT_KEY_SIZE = 2048


class JLS(object):
    def __init__(self, nusers: int, VE: VES = None) -> None:
        """
        Initializes a JLS instance.

        Args:
            nusers (int): The number of users in the system.
            VE (VES, optional): An instance of VES for vector encoding. Defaults to None.
        """
        super().__init__()
        self.nusers: int = nusers
        self.keysize: int = None
        self.ve: VES = VE

    def setup(
        self, lmbda: int = DEFAULT_KEY_SIZE
    ) -> Tuple[PublicParamJL, ServerKeyJL, Dict[int, UserKeyJL]]:
        """
        Performs the system setup to generate public parameters, server key, and user keys.

        Args:
            lmbda (int, optional): The security parameter. Defaults to DEFAULT_KEY_SIZE.

        Returns:
            Tuple[PublicParamJL, ServerKeyJL, Dict[int, UserKeyJL]]: A tuple containing public parameters,
            server key, and a dictionary of user keys.
        """
        self.keysize: int = lmbda

        p: Optional[int] = None
        q: Optional[int] = None
        n: Optional[int] = None
        n_len: int = 0
        while n_len != lmbda // 2:
            p = getprimeover(lmbda // 4)
            q = p
            while q == p:
                q = getprimeover(lmbda // 4)
            n = p * q
            n_len = n.bit_length()
        fdh: FDH = FDH(self.keysize, n * n)

        public_param: PublicParamJL = PublicParamJL(n, lmbda // 2, fdh.H)

        seed: random.Random = random.SystemRandom()
        s0: mpz = mpz(0)
        users: Dict[int, UserKeyJL] = {}

        for i in range(self.nusers):
            s: mpz = mpz(seed.getrandbits(2 * n_len))
            users[i] = UserKeyJL(public_param, s)
            s0 += s
        s0 = -s0
        server: ServerKeyJL = ServerKeyJL(public_param, s0)

        return public_param, server, users

    def protect(
        self,
        pp: PublicParamJL,
        sk_u: UserKeyJL,
        tau: int,
        x_u_tau: Union[int, List[int]],
    ) -> Union[EncryptedNumberJL, List[EncryptedNumberJL]]:
        """
        Protects a user's input by encrypting it.

        Args:
            pp (PublicParamJL): Public parameters.
            sk_u (UserKeyJL): User's key.
            tau (int): Time parameter.
            x_u_tau (Union[int, List[int]]): Input value(s) to protect.

        Returns:
            Union[EncryptedNumberJL, List[EncryptedNumberJL]]: The protected input(s).
        """
        assert isinstance(sk_u, UserKeyJL), "bad user key"
        assert sk_u.pp == pp, "bad user key"

        if isinstance(x_u_tau, list):
            x_u_tau = self.ve.encode(x_u_tau)
            return sk_u.encrypt(x_u_tau, tau)
        else:
            return sk_u.encrypt(x_u_tau, tau)

    def agg(
        self,
        pp: PublicParamJL,
        sk_0: ServerKeyJL,
        tau: int,
        list_y_u_tau: List[List[EncryptedNumberJL]],
    ) -> int:
        """
        Aggregates protected inputs and computes the sum.

        Args:
            pp (PublicParamJL): Public parameters.
            sk_0 (ServerKeyJL): Server key.
            tau (int): Time parameter.
            list_y_u_tau (List[List[EncryptedNumberJL]]): List of protected inputs.

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
            for y_u_tau in list_y_u_tau:
                assert len(list_y_u_tau[0]) == len(
                    y_u_tau
                ), "attempting to aggregate protected vectors of different sizes"

            y_tau = []
            for i in range(len(list_y_u_tau[0])):
                y_tau_i = list_y_u_tau[0][i]
                for y_u_tau in list_y_u_tau[1:]:
                    y_tau_i += y_u_tau[i]
                y_tau.append(y_tau_i)

            d = sk_0.decrypt(y_tau, tau, ttp=False)
            sum_x_u_tau: int = self.ve.decode(d)

        else:
            assert isinstance(list_y_u_tau[0], EncryptedNumberJL), "bad ciphertext"

            y_tau = list_y_u_tau[0]
            for y_u_tau in list_y_u_tau[1:]:
                y_tau += y_u_tau
            sum_x_u_tau: int = sk_0.decrypt(y_tau, tau)

        return sum_x_u_tau
