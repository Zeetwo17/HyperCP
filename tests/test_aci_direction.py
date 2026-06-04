import pytest
import torch
from hypercp.calibration.aci import ACI, ACIConfig

def test_err_above_target_alpha_decreases():
    torch.manual_seed(42)
    cfg = ACIConfig(alpha_target=0.10, eta=0.05, window=10)
    aci = ACI(cfg)
    
    q_cal = torch.zeros(10, 1, 2)
    q_cal[..., 0] = -1.0
    q_cal[..., 1] = 1.0
    y_cal = torch.zeros(10, 1)
    aci.fit_initial(q_cal, y_cal)
    
    aci.state.alpha_t = torch.tensor(0.10)
    
    q_pred = torch.zeros(1, 2)
    q_pred[..., 0] = -1.0
    q_pred[..., 1] = 1.0
    y_pred = torch.tensor([5.0]) # outside prediction interval -> err=1
    
    _, _, covered, _ = aci.step(q_pred, y_pred)
    assert not covered.item()
    
    new_alpha = aci.state.alpha_t.item()
    assert new_alpha < 0.10

def test_err_below_target_alpha_increases():
    torch.manual_seed(42)
    cfg = ACIConfig(alpha_target=0.10, eta=0.05, window=10)
    aci = ACI(cfg)
    
    q_cal = torch.zeros(10, 1, 2)
    q_cal[..., 0] = -1.0
    q_cal[..., 1] = 1.0
    y_cal = torch.zeros(10, 1)
    aci.fit_initial(q_cal, y_cal)
    
    aci.state.alpha_t = torch.tensor(0.10)
    
    q_pred = torch.zeros(1, 2)
    q_pred[..., 0] = -2.0
    q_pred[..., 1] = 2.0
    y_pred = torch.tensor([0.0]) # inside -> err=0
    
    _, _, covered, _ = aci.step(q_pred, y_pred)
    assert covered.item()
    
    new_alpha = aci.state.alpha_t.item()
    assert new_alpha > 0.10

def test_step_matches_gibbs_candes_formula():
    torch.manual_seed(42)
    cfg = ACIConfig(alpha_target=0.10, eta=0.05, window=10)
    aci = ACI(cfg)
    
    q_cal = torch.zeros(10, 1, 2)
    q_cal[..., 0] = -1.0
    q_cal[..., 1] = 1.0
    y_cal = torch.zeros(10, 1)
    aci.fit_initial(q_cal, y_cal)

    combinations = [
        (0.20, 1.0),
        (0.10, 0.0),
        (0.50, 1.0),
        (0.90, 0.0),
    ]

    for start_alpha, err in combinations:
        aci.state.alpha_t = torch.tensor(start_alpha)
        
        q_pred = torch.zeros(1, 2)
        q_pred[..., 0] = -1.0
        q_pred[..., 1] = 1.0
        
        y_val = 5.0 if err == 1.0 else 0.0
        y_pred = torch.tensor([y_val])
        
        aci.step(q_pred, y_pred)
        
        expected_alpha = start_alpha + cfg.eta * (cfg.alpha_target - err)
        assert torch.isclose(aci.state.alpha_t, torch.tensor(expected_alpha), atol=1e-5)

def test_convergence_iid():
    torch.manual_seed(42)
    n_streams = 50
    T_cal = 200
    cfg = ACIConfig(alpha_target=0.10, eta=0.05, window=T_cal)
    aci = ACI(cfg)
    
    K = 2
    quantiles = torch.tensor([0.05, 0.95])
    
    q_template = 0.8 * torch.distributions.Normal(0, 1).icdf(quantiles)
    q_pred_step = q_template.unsqueeze(0).expand(n_streams, K)
    
    y_cal = torch.randn(T_cal, n_streams)
    q_cal = q_pred_step.unsqueeze(0).expand(T_cal, n_streams, K)
    
    aci.fit_initial(q_cal, y_cal)
    
    T_test = 500
    for _ in range(T_test):
        y_t = torch.randn(n_streams)
        aci.step(q_pred_step, y_t)
        
    steady_state_cov = torch.tensor(aci.state.coverage_history[T_test//2:]).mean().item()
    assert 0.85 <= steady_state_cov <= 0.95

def test_negative_feedback_direction():
    torch.manual_seed(42)
    cfg = ACIConfig(alpha_target=0.10, eta=0.05, window=10)
    aci = ACI(cfg)
    
    q_cal = torch.zeros(10, 1, 2)
    q_cal[..., 0] = -1.0
    q_cal[..., 1] = 1.0
    y_cal = torch.zeros(10, 1)
    aci.fit_initial(q_cal, y_cal)
    
    aci.state.alpha_t = torch.tensor(0.10)
    
    alphas_overcoverage = []
    for _ in range(10):
        q_pred = torch.zeros(1, 2)
        q_pred[..., 0] = -1.0
        q_pred[..., 1] = 1.0
        y_pred = torch.tensor([0.0]) # err=0
        
        _, _, covered, _ = aci.step(q_pred, y_pred)
        assert covered.item()
        alphas_overcoverage.append(aci.state.alpha_t.item())
        
    for i in range(1, 10):
        assert alphas_overcoverage[i] > alphas_overcoverage[i-1]
        
    alphas_undercoverage = []
    for _ in range(10):
        q_pred = torch.zeros(1, 2)
        q_pred[..., 0] = -1.0
        q_pred[..., 1] = 1.0
        y_pred = torch.tensor([5.0]) # err=1
        
        _, _, covered, _ = aci.step(q_pred, y_pred)
        assert not covered.item()
        alphas_undercoverage.append(aci.state.alpha_t.item())
        
    for i in range(1, 10):
        assert alphas_undercoverage[i] <= alphas_undercoverage[i-1]
    # Verify alpha actually decreased at least once before hitting the clamp floor
    assert alphas_undercoverage[-1] < alphas_undercoverage[0]
