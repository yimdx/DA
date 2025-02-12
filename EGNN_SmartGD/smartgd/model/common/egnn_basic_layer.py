from .module_factory import ModuleFactory
from .norm_wrappers import NormWrapper
from .skip_connection import SkipConnection
from .egnn_clean import E_GCL

from dataclasses import dataclass
from typing import Optional
from attrs import define, frozen
import torch
from torch import nn, jit, FloatTensor, LongTensor
import torch_geometric as pyg


@define(kw_only=True, eq=False, repr=False, slots=False)
class EGNNBasicLayer(nn.Module):

    @dataclass(kw_only=True)
    class Params:
        in_dim: int
        out_dim: int
        hidden_dim: int
        edge_feat_dim: int

    @dataclass(kw_only=True)
    class Config:
        dense: bool = False
        bn: Optional[str] = "batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.0
        residual: bool = False
        aggr: str = "mean"
        root_weight: bool = True
        norm: bool = True

    layer_index: int
    params: Params
    config: Config = Config()

    def __attrs_post_init__(self):
        super().__init__()

        # Define flags
        self.with_dense: bool = self.config.dense
        self.with_bn: bool = self.config.bn is not None
        self.with_act: bool = self.config.act is not None
        self.with_dp: bool = self.config.dp > 0.0
        self.residual: bool = self.config.residual

        # Define nn modules
        self.e_gcl: nn.Module = E_GCL(
            self.params.in_dim, 
            self.params.out_dim,
            self.params.hidden_dim,
            edges_in_d = self.params.edge_feat_dim,
            act_fn = ModuleFactory(self.config.act)(),
            coords_agg="sum",
            residual=self.config.residual,
            normalize=True
        )
        self.dense: nn.Module = nn.Linear(self.params.out_dim, self.params.out_dim)
        self.bn: nn.Module = NormWrapper(ModuleFactory("pyg_batch_norm")(self.params.out_dim))
        self.act: nn.Module = ModuleFactory(self.config.act)()
        self.dp: nn.Module = nn.Dropout(self.config.dp)
        self.skip: SkipConnection = SkipConnection(in_dim=self.params.in_dim, out_dim=self.params.out_dim)
        self.bn2 = NormWrapper(ModuleFactory("pyg_batch_norm")(2))

    def forward(self, *,
                coords: FloatTensor,
                node_feat: FloatTensor,
                edge_feat: FloatTensor,
                edge_index: LongTensor,
                batch_index: LongTensor,
                num_sampled_nodes_per_hop: list[int],
                num_sampled_edges_per_hop: list[int]) -> tuple[FloatTensor, LongTensor, FloatTensor]:
        inputs = outputs = node_feat
        outputs, coords, edge_feat = self.e_gcl(h=outputs, edge_index=edge_index, coord=coords, edge_attr=edge_feat)
        if self.with_dense:
            outputs = self.dense(outputs)
        # if self.with_bn:
        outputs = self.bn(node_feat=outputs, batch_index=batch_index)
        coords = self.bn2(node_feat=coords, batch_index=batch_index)
        if self.with_act:
            outputs = self.act(outputs)
        if self.with_dp:
            outputs = self.dp(outputs)
        if self.residual:
            outputs = self.skip(block_input=inputs, block_output=outputs)
        return outputs, coords, edge_feat


EGNNBasicLayer.__annotations__.clear()
