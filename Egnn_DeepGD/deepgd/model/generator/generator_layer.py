from deepgd.constants import EPS
from ..common import NNConvLayer, NNConvBasicLayer, EdgeFeatureExpansion
from ..common.egnn_basic_layer import EGNNBasicLayer
from typing import Optional
from attrs import define, frozen
from dataclasses import dataclass

import torch
from torch import nn, jit, FloatTensor, LongTensor


@define(kw_only=True, eq=False, repr=False, slots=False)
class GeneratorLayer(nn.Module):

    @dataclass(kw_only=True)
    class Config:
        in_dim: int
        out_dim: int
        hidden_dim: int
        node_feat_dim: int
        edge_feat_dim: int

    @dataclass(kw_only=True)
    class EdgeNetConfig:
        width: int = 0
        depth: int = 0
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = "tanh"
        bn: Optional[str] = "batch_norm"
        dp: float = 0.0
        residual: bool = False

    @dataclass(kw_only=True)
    class GNNConfig:
        aggr: str = "mean"
        root_weight: bool = True
        dense: bool = False
        bn: Optional[str] = "batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.0
        norm: bool = True

    layer_index: int
    config: Config
    edge_net_config: EdgeNetConfig = EdgeNetConfig()
    gnn_config: GNNConfig = GNNConfig()
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions(),
    eps: float = EPS

    def __attrs_post_init__(self):
        super().__init__()

        self.edge_feat_provider: EdgeFeatureExpansion = EdgeFeatureExpansion(
            config=EdgeFeatureExpansion.Config(
                node_feat_dim=self.config.node_feat_dim,
                edge_attr_dim=self.config.edge_feat_dim
            ),
            expansions=self.edge_feat_expansion,
            eps=self.eps
        )

        self.gnn_layer: EGNNBasicLayer = EGNNBasicLayer(
            layer_index=self.layer_index,
            params=EGNNBasicLayer.Params(
                in_dim=self.config.in_dim,
                out_dim=self.config.out_dim,
                hidden_dim=self.config.hidden_dim,
                edge_feat_dim=self.edge_feat_provider.get_feature_channels()
            ),
            config=EGNNBasicLayer.Config(
                dense=self.gnn_config.dense,
                bn=self.gnn_config.bn,
                act=self.gnn_config.act,
                dp=self.gnn_config.dp,
                residual=False,
                aggr=self.gnn_config.aggr,
                root_weight=self.gnn_config.root_weight,
                norm=self.gnn_config.norm
            ),
        )

    def forward(self, *,
                coords: FloatTensor,
                node_feat: FloatTensor,
                edge_feat: FloatTensor,
                edge_index: LongTensor,
                batch_index: LongTensor,
                num_sampled_nodes_per_hop: list[int],
                num_sampled_edges_per_hop: list[int]) -> tuple[FloatTensor, LongTensor, FloatTensor]:
        return self.gnn_layer(
            coords=coords,
            node_feat=node_feat,
            edge_feat=self.edge_feat_provider(
                node_feat=node_feat,
                edge_index=edge_index,
                edge_attr=edge_feat
            ),
            edge_index=edge_index,
            batch_index=batch_index,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop
        )


GeneratorLayer.__annotations__.clear()
