from math import log2
from random import randint

import pytest

from src.buildingblocks import JLS, VES, EncryptedNumberJL, PublicParamJL, ServerKeyJL


class TestJLS:
    @pytest.fixture
    def jls_instance(self) -> JLS:
        nusers = 100  # You can adjust the number of users as needed
        return JLS(nusers)

    def test_setup(self, jls_instance):
        lmbda = 2048  # You can adjust the key size as needed
        public_param, server_key, user_keys = jls_instance.setup(lmbda)
        assert isinstance(public_param, PublicParamJL)
        assert isinstance(server_key, ServerKeyJL)
        assert isinstance(user_keys, dict)
        assert len(user_keys) == jls_instance.nusers

    def test_protect(self, jls_instance):
        lmbda = 2048
        public_param, server_key, user_keys = jls_instance.setup(lmbda)
        user_key = user_keys[0]  # Choose a user for testing

        tau = 1  # You can adjust the tau value as needed
        x_u_tau = 42  # Replace with your input data
        encrypted_x = jls_instance.protect(public_param, user_key, tau, x_u_tau)

        assert isinstance(encrypted_x, EncryptedNumberJL)

    def test_agg(self, jls_instance):
        lmbda = 2048
        public_param, server_key, user_keys = jls_instance.setup(lmbda)
        tau = 1  # You can adjust the tau value as needed
        x_u_tau = randint(0, 2 ** (lmbda // 2 - log2(jls_instance.nusers) - 1))
        print(x_u_tau.bit_length())
        print((x_u_tau * jls_instance.nusers).bit_length())
        list_y_u_tau = [
            jls_instance.protect(public_param, user_key, tau, x_u_tau)
            for user_key in user_keys.values()
        ]
        result = jls_instance.agg(public_param, server_key, tau, list_y_u_tau)

        assert result == x_u_tau * jls_instance.nusers


class TestVES:
    @pytest.fixture
    def ves_instance(self):
        ptsize = 64  # You can adjust these values as needed
        addops = 4
        valuesize = 8
        vectorsize = 16
        return VES(ptsize, addops, valuesize, vectorsize)

    def test_encode_decode(self, ves_instance):
        V = [1, 2, 3, 4, 5, 6, 7, 8]
        E = ves_instance.encode(V)
        decoded_V = ves_instance.decode(E)

        assert V == decoded_V


if __name__ == "__main__":
    pytest.main()

if __name__ == "__main__":
    pytest.main()
