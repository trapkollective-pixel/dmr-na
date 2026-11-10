import torch
import torch.nn as nn

class HarmonicDigitalRootGate(nn.Module):
    """
    Continuous harmonic approximation of modular digital root reduction.
    Surrogate Fourier series mapping inputs onto the modular residue manifold.
    """
    def __init__(self, harmonics: int = 4, scale: float = 1.0):
        super().__init__()
        self.harmonics = harmonics
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x = self.scale * x - 1.0
        h_val = torch.full_like(x, 5.0)
        for k in range(1, self.harmonics + 1):
            coeff = 4.0 / (k * torch.pi)
            freq = (2.0 * torch.pi * k) / 9.0
            h_val = h_val - coeff * torch.sin(freq * scaled_x)
        return h_val / 9.0

class DualMirroredRecursiveCell(nn.Module):
    """
    Dual-Mirrored Recurrent Cell executing cross-coupled primary and 
    mirrored updates with harmonic digital root stabilization.
    """
    def __init__(self, dim: int, coupling_gamma: float = 0.35, max_recurrence: int = 6):
        super().__init__()
        self.dim = dim
        self.gamma = coupling_gamma
        self.max_recurrence = max_recurrence
        
        self.w_primary = nn.Linear(dim, dim, bias=False)
        self.u_primary = nn.Linear(dim, dim, bias=False)
        self.w_mirror = nn.Linear(dim, dim, bias=False)
        self.u_mirror = nn.Linear(dim, dim, bias=False)
        
        self.root_gate_p = HarmonicDigitalRootGate(harmonics=4, scale=1.0)
        self.root_gate_m = HarmonicDigitalRootGate(harmonics=4, scale=1.0)

    def forward(self, x: torch.Tensor, h_p: torch.Tensor, h_m: torch.Tensor):
        p_in = self.w_primary(x)
        m_in = self.w_mirror(x)
        curr_p, curr_m = h_p, h_m
        
        for _ in range(self.max_recurrence):
            p_gate = self.root_gate_p(curr_p)
            m_gate = self.root_gate_m(curr_m)
            
            p_next = torch.tanh(p_in + self.u_primary(curr_p) + self.gamma * (curr_m * m_gate))
            m_next = torch.tanh(m_in + self.u_mirror(curr_m) + self.gamma * (curr_p * p_gate))
            
            delta = torch.norm(p_next - m_next, p=2, dim=-1).mean()
            curr_p, curr_m = p_next, m_next
            
            if delta.item() < 1e-4:
                break
                
        return curr_p, curr_m, delta
