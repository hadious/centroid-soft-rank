import math
import torch
from torch.nn import functional as F
from losses import ssl_loss
from losses.rank_only import centroid_soft_rank, soft_rank

def test_normalized_centroid_matches_definition_and_gradients():
    torch.manual_seed(5)
    z=torch.randn(8,5,dtype=torch.float64,requires_grad=True)
    v=torch.randn_like(z,requires_grad=True)
    c=F.normalize((F.normalize(z,dim=1)+F.normalize(v,dim=1))/2,dim=1)
    expected=1-11/5*soft_rank(c/math.sqrt(8),2.)
    actual,_=ssl_loss(z,v,'centroid_normalized',kappa=10.)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(v.grad).all()

def test_shuffle_uses_wrong_pairs_preserves_rng_and_replays():
    torch.manual_seed(17)
    z=torch.randn(8,5,requires_grad=True);v=torch.randn_like(z,requires_grad=True)
    generator=torch.Generator().manual_seed(torch.initial_seed())
    order=torch.randperm(8,generator=generator)
    perm=torch.empty_like(order);perm[order]=order.roll(1)
    assert (perm != torch.arange(8)).all()
    expected,_=ssl_loss(z,v[perm],'centroid_soft',kappa=10.)
    centroid_soft_rank.shuffle_step=0
    state=torch.get_rng_state().clone()
    actual,_=ssl_loss(z,v,'centroid_shuffled',kappa=10.)
    torch.testing.assert_close(actual,expected)
    assert torch.equal(state,torch.get_rng_state())
    centroid_soft_rank.shuffle_step=0
    replay,_=ssl_loss(z,v,'centroid_shuffled',kappa=10.)
    torch.testing.assert_close(actual,replay)
    actual.backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(v.grad).all()
