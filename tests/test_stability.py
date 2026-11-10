import torch
from dmr_na.modules import HarmonicDigitalRootGate, DualMirroredRecursiveCell
from dmr_na.model import DMRNetwork

def test_harmonic_digital_root_boundedness():
    gate = HarmonicDigitalRootGate(harmonics=4)
    x = torch.linspace(-50.0, 50.0, steps=2000)
    out = gate(x)
    assert not torch.isnan(out).any(), "Harmonic gate produced NaN"
    assert out.min() >= -0.2 and out.max() <= 1.2, "Gate output exceeded bounded range"

def test_cell_contraction_convergence():
    cell = DualMirroredRecursiveCell(dim=32, coupling_gamma=0.35, max_recurrence=10)
    x = torch.randn(8, 32)
    h_p = torch.zeros(8, 32)
    h_m = torch.zeros(8, 32)
    p_next, m_next, delta = cell(x, h_p, h_m)
    assert not torch.isnan(p_next).any() and not torch.isnan(m_next).any()
    assert delta.item() < 10.0, "State divergence detected in coupled fields"

def test_full_sequence_forward():
    model = DMRNetwork(input_dim=4, hidden_dim=16, output_dim=1)
    seq = torch.randn(4, 25, 4)
    out = model(seq)
    assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"
