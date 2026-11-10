import torch
import torch.nn as nn
from .modules import DualMirroredRecursiveCell

class DMRNetwork(nn.Module):
    """
    Sequence model wrapper integrating projection layers, 
    the dual-mirrored recurrence cell, and output readout.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 coupling_gamma: float = 0.35, max_recurrence: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.proj_p = nn.Linear(input_dim, hidden_dim)
        self.proj_m = nn.Linear(input_dim, hidden_dim)
        self.cell = DualMirroredRecursiveCell(dim=hidden_dim, coupling_gamma=coupling_gamma, max_recurrence=max_recurrence)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_seq.shape
        h_p = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        h_m = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            p_embed = self.proj_p(x_t)
            m_embed = self.proj_m(x_t)
            h_p, h_m, _ = self.cell(p_embed + m_embed, h_p, h_m)
            
        fused_state = torch.cat([h_p, h_m], dim=-1)
        return self.head(fused_state)
