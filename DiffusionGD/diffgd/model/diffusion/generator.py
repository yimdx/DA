import torch.nn as nn
import torch
from ..common.egnn_basic_layer import EGNNBasicLayer
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, edge_feat_dim=2, timesteps=100, device = 'cpu'):
        super().__init__()
        self.timesteps = timesteps
        self.betas = self.linear_beta_schedule(timesteps).to(device)
        self.alphas = (torch.tensor(1.0) - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        
        self.params = EGNNBasicLayer.Params(in_dim=in_dim*2, out_dim=in_dim*2, hidden_dim=in_dim*2, edge_feat_dim=edge_feat_dim),
        self.config = EGNNBasicLayer.Config(
                    dense = False,
                    bn = "pyg_batch_norm",
                    act= "leaky_relu",
                    dp = 0.0,
                    residual = False,
                    aggr = "mean",
                    root_weight = True,
                    norm = True
        ),

        self.layers = nn.ModuleList()
        for index in range(num_layers):
            self.layers.append(
                EGNNBasicLayer(
                    layer_index=index,
                    params=EGNNBasicLayer.Params(in_dim=in_dim*2, out_dim=in_dim*2, hidden_dim=in_dim*2, edge_feat_dim=edge_feat_dim),
                    config=EGNNBasicLayer.Config(
                            dense = False,
                            bn = "pyg_batch_norm",
                            act= "leaky_relu",
                            dp = 0.0,
                            residual = False,
                            aggr = "mean",
                            root_weight = True,
                            norm = True
                    ),
                ))
        self.time_embed = nn.Embedding(timesteps, in_dim)
        self.readout = nn.Linear(out_dim, 2)


    def linear_beta_schedule(self, num_timesteps):
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps)
    
    def forward(self, coords, node_feat, edge_index, edge_attr, batch_index, t):
        t = torch.full((node_feat.shape[0],), t, dtype=torch.long, device=next(self.parameters()).device)
        t_embed = self.time_embed(t) 
        output = torch.cat([node_feat, t_embed], dim=-1)
        # print(coords.shape)
        # print(output.shape)
        # print(edge_index.shape)
        # print(edge_attr.shape)
        for layer in self.layers:
            output, coords, edge_attr = layer(node_feat=output, coords=coords, edge_index=edge_index, edge_feat=edge_attr, batch_index=batch_index)
        pred_noise = self.readout(output)
        return coords, pred_noise
        noise = torch.randn_like(coords) * (t / self.timesteps)
        coords = coords + noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    

    def p_losses(self, x_start, node_feat, edge_index, edge_attr, batch_index, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        _, pred_noise = self.forward(
            coords=x_noisy,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch_index=batch_index,
            t=t,
        )
        loss = F.mse_loss(pred_noise, noise)
        return loss

    

    def p_sample(self, x, t, node_feat, edge_index, edge_attr, batch_index):
        """Reverse diffusion: Denoise coordinates by one step."""
        with torch.no_grad():
            _, pred_noise = self.forward(x, node_feat, edge_index, edge_attr, batch_index, t)

            alpha_t = self.alphas[t]       # [batch_size * num_nodes, 1]
            beta_t = self.betas[t]           # [batch_size * num_nodes, 1]
            alphas_cumprod_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            # print(t)
            # print(pred_noise.shape)
            # print(alpha_t.shape)
            # print(beta_t.shape)
            # print(alphas_cumprod_t.shape)
            # print(sqrt_alpha_t.shape)
            x_prev = (x - (beta_t / sqrt_alpha_t) * pred_noise) / torch.sqrt(1 - alphas_cumprod_t)
            
            sigma_t = torch.sqrt(beta_t)
            if t>0:
                x_prev += sigma_t * torch.randn_like(x_prev)
            
            return x_prev
        
    def sample(self, node_feat, edge_index, edge_attr, batch_index, num_samples=1):
        device = next(self.parameters()).device
        x = torch.randn((node_feat.shape[0], 2), device=device)
        
        for t_step in reversed(range(self.timesteps)):
            x = self.p_sample(x, t_step, node_feat, edge_index, edge_attr, batch_index)
        return x
        
if __name__ == "__main__":
    import numpy as np
    import torch
    model = DiffusionModel(in_dim=16, out_dim=16, num_layers=4, edge_feat_dim=2, timesteps=1000, device='cpu').to('cpu')
    init_pos = torch.rand_like((16, 2))
    node_feats = torch.rand_like(16, 16)
    perm_index = torch.ones(2, 240)
    perm_index = torch.ones(2, 240)
    loss = model.p_losses(
        x_start=init_pos,
        node_feat=node_feats,
        edge_index=perm_index,
        edge_attr=torch.rand_like(240, 2),
        batch_index=0,
        t=0
    )