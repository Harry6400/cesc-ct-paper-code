import pytest
import torch
import torch.nn.functional as F
from cesc.ops import patches,patch_adjoint,pool,pool_adjoint,low_rank_solve,low_rank_logdet,LocalMetric

@pytest.mark.parametrize('size',[(4,4),(8,12),(16,16)])
def test_patch_adjoint(size):
    x=torch.randn(2,1,*size,dtype=torch.float64);v=torch.randn_like(patches(x))
    torch.testing.assert_close((patches(x)*v).sum(),(x*patch_adjoint(v,size)).sum())

@pytest.mark.parametrize('factor',[1,2,4])
def test_pool_adjoint(factor):
    x=torch.randn(2,1,16,12,dtype=torch.float64);v=torch.randn_like(pool(x,factor))
    torch.testing.assert_close((pool(x,factor)*v).sum(),(x*pool_adjoint(v,factor)).sum())

@pytest.mark.parametrize('rank',[1,2,3,4])
def test_woodbury_matches_dense(rank):
    d=torch.rand(2,7,9,dtype=torch.float64)+.1;u=torch.randn(2,7,9,rank,dtype=torch.float64)*.4
    v=torch.randn_like(d);m=torch.diag_embed(d)+u@u.transpose(-2,-1)
    torch.testing.assert_close(low_rank_solve(d,u,v),torch.linalg.solve(m,v.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(low_rank_logdet(d,u),torch.linalg.slogdet(m).logabsdet)

@pytest.mark.parametrize('mode',['full','diagonal'])
def test_metric_symmetry_psd_bound(mode):
    size=(8,8);d=torch.rand(2,36,9,dtype=torch.float64)+.1;u=torch.randn(2,36,9,2,dtype=torch.float64)*.1
    m=LocalMetric(d,u,size,mode);x=torch.randn(2,1,*size,dtype=torch.float64);y=torch.randn_like(x)
    torch.testing.assert_close((x*m.apply(y)).sum(),(y*m.apply(x)).sum())
    assert (m.quadratic(x)>=0).all()
    lhs=(x*m.apply(x)).flatten(1).sum(-1);rhs=m.upper_bound().flatten()*x.square().flatten(1).sum(-1)
    assert (lhs<=rhs+1e-8).all()

def test_diagonal_preserves_marginals_and_bound():
    d=torch.rand(1,36,9)+.1;u=torch.randn(1,36,9,2)*.1
    f=LocalMetric(d,u,(8,8));q=LocalMetric(d,u,(8,8),'diagonal');v=torch.randn_like(d)
    torch.testing.assert_close(q.solve(v),v/(d+u.square().sum(-1)))
    torch.testing.assert_close(f.upper_bound(),q.upper_bound(),rtol=0,atol=0)

def test_solver_gradcheck():
    d=(torch.rand(1,2,9,dtype=torch.float64)+.5).requires_grad_()
    u=(torch.randn(1,2,9,2,dtype=torch.float64)*.1).requires_grad_()
    v=torch.randn(1,2,9,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(low_rank_solve,(d,u,v),fast_mode=True)

def test_invalid_patch_shape():
    with pytest.raises(ValueError):patches(torch.randn(1,2,8,8))
