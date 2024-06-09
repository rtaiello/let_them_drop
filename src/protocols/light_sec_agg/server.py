"""
Implementation LightSecAgg protocol [1] https://arxiv.org/pdf/2109.14236.pdf.
"""
from collections import defaultdict
from math import ceil
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from ...buildingblocks import LCC, add_vectors, get_field, sub_vectors
from ...buildingblocks.key_agreement import KAS


class LightSecAggServer:
    dimension: int = 1000
    nclients: int = 10
    valuesize: int = 16
    key_size: int = 64
    threshold: int = ceil(2 * nclients / 3)
    clients: List[int] = [i + 1 for i in range(nclients)]
    lcc: LCC = LCC(key_size)

    def __init__(self) -> None:
        """
        Initialize a LightSecAggServer instance.

        Returns:
            None
        """
        super().__init__()
        self.ckeys: Dict[Any, Any] = {}
        self.clients: List[Any] = []
        self.clients_on: List[Any] = []
        self.key_shares: Dict[Any, Any] = {}
        self.x: List[Any] = []
        self.ka_channel: KAS = KAS()
        self.y: List[Any] = []

    @staticmethod
    def set_scenario(
        dimension: int, valuesize: int, keysize: int, threshold: int, nclients: int
    ) -> None:
        """
        Set the scenario parameters for LightSecAggServer.

        Args:
            dimension (int): The dimension.
            valuesize (int): The valuesize.
            keysize (int): The keysize.
            threshold (int): The threshold.
            nclients (int): The number of clients.

        Returns:
            None
        """
        LightSecAggServer.dimension = dimension
        LightSecAggServer.valuesize = valuesize
        LightSecAggServer.nclients = nclients
        LightSecAggServer.nclients_target_on = threshold + 1
        LightSecAggServer.threshold = threshold
        LightSecAggServer.d_encoded = ceil(
            LightSecAggServer.dimension
            / (LightSecAggServer.nclients_target_on - LightSecAggServer.threshold)
        )
        LightSecAggServer.keysize = keysize
        LightSecAggServer.threshold = threshold
        LightSecAggServer.clients = [i + 1 for i in range(nclients)]
        LightSecAggServer.lcc = LCC(keysize)
        LightSecAggServer.field = get_field(keysize)

    def setup_register(self, alldhpkc: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Perform the setup registration step.

        Args:
            alldhpkc (Dict[Any, Any]): A dictionary of all clients' public keys.

        Returns:
            Dict[Any, Any]: A dictionary of all clients' public keys.
        """
        assert len(alldhpkc.keys()) >= LightSecAggServer.threshold
        return alldhpkc

    def distribute_local_masks(
        self, alleshares: Dict[Any, Dict[Any, Any]]
    ) -> Dict[Any, Dict[Any, Any]]:
        """
        Distribute local masks.

        Args:
            alleshares (Dict[Any, Dict[Any, Any]]): A dictionary of local mask shares.

        Returns:
            Dict[Any, Dict[Any, Any]]: A dictionary of distributed local masks.
        """
        eshares: Dict[Any, Dict[Any, Any]] = defaultdict(dict)
        for u in alleshares.keys():
            for v in alleshares[u].keys():
                eshares[v][u] = alleshares[u][v]
        return eshares

    def online_encrypt(self, all_y: Dict[Any, Any]) -> List[Any]:
        """
        Perform online encryption.

        Args:
            all_y (Dict[Any, Any]): A dictionary of encrypted data.

        Returns:
            List[Any]: A list of clients who participated in encryption.
        """
        self.clients_on = list(all_y.keys())
        aggregated: List[Any] = [self.field(0)] * LightSecAggServer.dimension
        for y in all_y.values():
            aggregated = add_vectors(aggregated, y)
        self.y = aggregated
        return self.clients_on

    def aggregate(self, all_sum_encoded_mask: Dict[Any, Any]) -> List[Any]:
        """
        Perform aggregation.

        Args:
            all_sum_encoded_mask (Dict[Any, Any]): A dictionary of sum encoded masks.

        Returns:
            List[Any]: A list of aggregated data.
        """
        field = LightSecAggServer.field
        nclients = LightSecAggServer.nclients
        dimension = LightSecAggServer.dimension
        d_encoded = LightSecAggServer.d_encoded
        nclients_on = len(self.clients_on)

        alpha_s: List[Any] = [field(i + 1) for i in range(nclients_on)]
        beta_s: List[Any] = [field(i + (nclients + 1)) for i in range(nclients_on)]
        all_sum_encoded_mask = [
            list(all_sum_encoded_mask[key]) for key in all_sum_encoded_mask.keys()
        ]
        all_sum_encoded_mask[: LightSecAggServer.nclients_target_on]
        decoded_mask = LightSecAggServer.lcc.decoding_with_points(
            all_sum_encoded_mask, alpha_s, beta_s
        )
        aggregate_mask = np.reshape(
            decoded_mask, (nclients_on * d_encoded, 1)
        ).flatten()
        aggregate_mask = aggregate_mask[:dimension].tolist()
        aggregated = sub_vectors(self.y, aggregate_mask)
        return aggregated
