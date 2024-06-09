"""
Code implemeted according to the paper repository:
    https://github.com/LightSecAgg/MLSys2022_anonymous
"""
from collections import defaultdict
from math import ceil
from typing import Dict, List

import numpy as np

from .utils import add_vectors, create_mask, get_field, multiply_vector_by_scalar


class LCC:
    def __init__(self, bitlength: int):
        super().__init__()
        self.field = get_field(bitlength)
        self.bitlength = bitlength

    def _lagrange_encoding_matrix(self, alpha_s, beta_s):
        """
        Lagrange encoding matrix described in https://arxiv.org/pdf/1806.00939.pdf Equation (2)
        """
        num_alpha = len(alpha_s)
        num_beta = len(beta_s)
        encoding_matrix = defaultdict(dict)
        den_s = []
        num_s = []

        for i in range(num_beta):
            b_i = beta_s[i]
            den = self.field(1)
            for b_l in beta_s:
                if b_i != b_l:
                    v = b_i - b_l
                    den = den * v
            den_s.append(den)

        for j in range(num_alpha):
            a_j = alpha_s[j]
            num = self.field(1)
            for b_l in beta_s:
                v = a_j - b_l
                num = num * v
            num_s.append(num)

        for i in range(num_beta):
            for j in range(num_alpha):
                den = (alpha_s[j] - beta_s[i]) * den_s[i]
                encoding_matrix[j][i] = num_s[j] * den.inverse()
        return encoding_matrix

    def lcc_encoding_with_points(self, xs, alpha_s, beta_s):
        xs_encoded = defaultdict(list)
        encoding_matrix = self._lagrange_encoding_matrix(alpha_s, beta_s)

        for i in range(len(alpha_s)):
            tmp_sum = [self.field(0)] * len(xs[0])
            for j in range(len(beta_s)):
                tmp_sum = add_vectors(
                    tmp_sum, multiply_vector_by_scalar(xs[j], encoding_matrix[i][j])
                )
            xs_encoded[i] = tmp_sum
        return xs_encoded

    def decoding_with_points(self, x_encoded, alpha_s, beta_s) -> List[list]:
        decoding_matrix = self._lagrange_encoding_matrix(beta_s, alpha_s)
        xs_decoded = defaultdict(list)
        for i in range(len(alpha_s)):
            tmp_sum = [self.field(0)] * len(x_encoded[0])
            for j in range(len(beta_s)):
                tmp_sum = add_vectors(
                    tmp_sum,
                    multiply_vector_by_scalar(x_encoded[j], decoding_matrix[i][j]),
                )
            xs_decoded[i] = tmp_sum
        return list(xs_decoded.values())

    def mask_encoding(
        self,
        dimension: int,
        nclients: int,
        nclients_target_on: int,
        threshold: int,
        xs: List[int],
    ):
        alpha_s = [self.field(i + 1) for i in range(nclients)]
        beta_s = [self.field(i + (nclients + 1)) for i in range(nclients_target_on)]

        nrows = (nclients_target_on - threshold) + 1
        ncols = ceil(dimension / (nclients_target_on - threshold))
        """
        from paper [1],Offline encoding and sharing of local masks, 
        partitions x_s in nclients_on - threshold + 1 sub-masks
        """
        if len(xs) != (ncols * nrows) and len(xs) < (ncols * nrows):
            xs = np.pad(
                xs,
                (0, (ncols * nrows) - len(xs)),
                "constant",
                constant_values=self.field(0),
            )
        if len(xs) > (ncols * nrows):
            xs = xs[: (ncols * nrows)]
            xs = np.array(xs)

        xs = xs.reshape(nrows, ncols)
        nrows_n_i = threshold - 1
        n_i = create_mask(self.bitlength, ncols * nrows_n_i)
        n_i = np.reshape(n_i, (nrows_n_i, ncols))

        assert len(n_i) + len(xs) == nclients_target_on

        lcc_in = np.concatenate((xs, n_i), axis=0)
        lcc_in = lcc_in.tolist()
        xs_encoded = self.lcc_encoding_with_points(lcc_in, alpha_s, beta_s)
        return xs_encoded
