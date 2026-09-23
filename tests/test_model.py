import copy
import pytest
import torch
from cesc.model import CESC,context_tensor
from cesc.statistics import statistics_loss
from cesc.losses import reconstruction_loss
from cesc.ops import LocalMetric

@pytest.mark.parametrize('variant',['full','diagonal','plain','b0'])
def test_initial_identity(cfg,variant):
    m=CESC(cfg,variant);c=torch.randn(2,1,16,16)*10+1000;b=c+torch.randn_like(c)
    with torch.no_grad():p=m(c,b)
    torch.testing.assert_close(p,b,rtol=0,atol=0)

@pytest.mark.parametrize('variant',['full','diagonal','plain'])
def test_alpha_zero_identity_after_learning(cfg,variant):
    m=CESC(cfg,variant).set_stage('correction')
    c=torch.randn(2,1,16,16)+1000;b=c+2;t=b+1
    loss=reconstruction_loss(m(c,b),t,cfg.s_ct);loss.backward()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001);opt.step()
    assert not torch.equal(m(c,b),b)
    torch.testing.assert_close(m(c,b,alpha=0),b,rtol=0,atol=0)

def test_isolated_constructor_rng(cfg):
    torch.manual_seed(100);state=torch.get_rng_state().clone();CESC(cfg)
    assert torch.equal(state,torch.get_rng_state())

def test_full_diagonal_same_initial_states(cfg):
    f=CESC(cfg,'full');d=CESC(cfg,'diagonal')
    for k,v in f.state_dict().items():torch.testing.assert_close(v,d.state_dict()[k],rtol=0,atol=0)

def test_statistics_lowrank_gradient_nonzero(cfg):
    m=CESC(cfg).set_stage('statistics');x=torch.randn(2,1,16,16)+1000;b=x+.5
    met=m.statistic_metric(x,b);loss,_=statistics_loss(met,torch.randn_like(x)*.1)
    loss.backward();g=m.statistics.head.weight.grad[9:]
    assert g.abs().sum()>0 and torch.isfinite(g).all()

def test_stats_frozen_in_stage_two(cfg):
    m=CESC(cfg).set_stage('correction');x=torch.randn(2,1,16,16)+1000;b=x+.5
    reconstruction_loss(m(x,b),b+1,cfg.s_ct).backward()
    assert all(p.grad is None and not p.requires_grad for p in m.statistics.parameters())
    assert not m.statistics.training
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.correction.parameters())

def test_potential_gradient_matches_autograd(cfg):
    m=CESC(cfg).double();c=torch.randn(2,3,8,8,dtype=torch.float64)
    pot=m.correction.potential
    with torch.no_grad():
        for h in pot.target_heads:h.weight.normal_(0,.05)
    state=pot.prepare(c);d=torch.randn(2,1,8,8,dtype=torch.float64,requires_grad=True)*.1
    actual=torch.autograd.grad(pot.value(d,state).sum(),d,create_graph=True)[0]
    torch.testing.assert_close(actual,pot.gradient(d,state),rtol=1e-9,atol=1e-10)

@pytest.mark.parametrize('mode',['full','diagonal'])
def test_total_energy_gradient_and_descent(cfg,mode):
    m=CESC(cfg).double();c=torch.randn(2,3,8,8,dtype=torch.float64)
    met=m.statistics(c,mode).detach();p=m.correction.potential
    with torch.no_grad():
        for h in p.target_heads:h.weight.normal_(0,.1);h.bias.fill_(.2)
    state=p.prepare(c);d=(torch.randn(2,1,8,8,dtype=torch.float64)*.1).requires_grad_()
    analytic=met.apply(d)+cfg.anchor*d+p.gradient(d,state)
    exact=torch.autograd.grad(m.correction.energy(d,met,state).sum(),d)[0]
    torch.testing.assert_close(exact,analytic,rtol=1e-9,atol=1e-10)
    _,trace=m.correction(c,met,return_trace=True)
    assert len(trace)==cfg.steps+1
    for a,b in zip(trace,trace[1:]):assert torch.all(b<=a+1e-9)

def test_no_target_argument_in_forward(cfg):
    import inspect
    assert 'target' not in str(inspect.signature(CESC.forward))

def test_input_requires_grad_rejected(cfg):
    c=torch.randn(1,1,8,8,requires_grad=True);b=c.detach()
    with pytest.raises(ValueError):context_tensor(c,b,cfg)

def test_bad_context_size(cfg):
    m=CESC(cfg)
    with pytest.raises(ValueError):m(torch.zeros(1,1,10,10),torch.zeros(1,1,10,10))

@pytest.mark.parametrize('alpha',[-1,2])
def test_bad_alpha(cfg,alpha):
    m=CESC(cfg)
    with pytest.raises(ValueError):m(torch.zeros(1,1,8,8),torch.zeros(1,1,8,8),alpha=alpha)

def test_loss_formula():
    p=torch.tensor([1.,3.]);t=torch.zeros(2)
    torch.testing.assert_close(reconstruction_loss(p,t,2.),torch.tensor(1.125))
