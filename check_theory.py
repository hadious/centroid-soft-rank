"""Float64 checks of CSR identities, gradients, certificates and counterexamples."""
import json
import torch
torch.set_num_threads(4)

def theory_checks():
    torch.manual_seed(7281)
    records=[]
    for b,d in [(2,1),(2,5),(3,3),(5,2),(7,11),(16,8)]:
        for kind in ["random","aligned","constant","antipodal","zero","floor"]:
            u=torch.randn(b,d,dtype=torch.float64)
            u=u/u.norm(dim=1,keepdim=True)
            v=torch.randn_like(u);v=v/v.norm(dim=1,keepdim=True)
            if kind=="aligned":v=u.clone()
            if kind=="constant":u=u[0:1].repeat(b,1);v=u.clone()
            if kind=="antipodal":v=-u
            if kind=="zero":u.zero_();v.zero_()
            if kind=="floor":u[0]*=.1;v[-1]*=.2
            c=(u+v)/2;g=c.T@c/b;r=min(b,d)
            # Independent singular-value and solve expressions.
            eigen=torch.linalg.svdvals(c/b**.5).square()
            for k in [.1,1.,10.,1e6]:
                tau=k/r;a=(1+k)/r;T=float(c.square().sum()/b);q=T/r
                score=a*float((eigen/(eigen+tau)).sum());loss=1-score
                solved=1-a*float(torch.trace(torch.linalg.solve(g+tau*torch.eye(d,dtype=torch.float64),g)))
                ap=k*(1-T)/(T+k)
                dp=float(a*tau/(q+tau)**2*((eigen-q)**2/(eigen+tau)).sum())
                err=abs(loss-ap-dp)
                assert err<1e-10 and abs(loss-solved)<1e-10,(b,d,kind,k,err,loss,solved)
                assert -.000000001<=loss<=1.000000001
                if loss<1:
                    eta=max(0.,loss);bound=4*eta*(1+k)/(k+eta)
                    assert float((u-v).square().sum()/b)<=bound+1e-8
                if kind=="constant":assert abs(loss-(r-1)/(r+k))<1e-9
                if kind=="antipodal":assert abs(loss-1)<1e-12
                if r>=2:
                    asymptotic=max(1/k,(r-1)/(r+k))
                    assert abs(loss-(1-T))<=asymptotic+1e-8
                records.append(dict(batch=b,dimension=d,kind=kind,kappa=k,decomposition_error=err))
    return {"cases":len(records),"maximum_decomposition_error":max(x["decomposition_error"] for x in records),"records":records}


def loss(c,k):
    b,d=c.shape;r=min(b,d);g=c.T@c/b
    return 1-(1+k)/r*torch.trace(torch.linalg.solve(g+k/r*torch.eye(d,dtype=c.dtype),g))

def additional_theory():
    torch.manual_seed(20261022);cases=[]
    for d in [2,3,7,16]:
        e=torch.eye(d,dtype=torch.float64);other=e.roll(1,0)
        for s in [1e-10,.01,.25,.5,1.]:
            u=s**.5*e+(1-s)**.5*other;v=s**.5*e-(1-s)**.5*other
            c=(u+v)/2;normalized=torch.nn.functional.normalize(c,dim=1,eps=1e-6)
            for k in [.1,1.,10.,100.]:
                ell=float(loss(c,k));modified=float(loss(normalized,k))
                assert abs(ell-k*(1-s)/(s+k))<1e-12
                assert abs(modified)<1e-12
                assert abs(float((u-v).square().sum()/d)-4*(1-s))<1e-12
                cases.append(dict(type='renormalization',d=d,s=s,kappa=k,error=abs(modified)))
    for b,d in [(2,5),(5,2),(4,4)]:
        for k in [.1,1.,10.,100.]:
            c=torch.randn(b,d,dtype=torch.float64,requires_grad=True)
            ell=loss(c,k);grad,=torch.autograd.grad(ell,c)
            g=c.T@c/b;r=min(b,d);a=(1+k)/r;tau=k/r;inv=torch.linalg.inv(g+tau*torch.eye(d,dtype=c.dtype))
            analytic=-2*a*tau/b*c@inv@inv
            error=float((grad-analytic).abs().max().detach());assert error<1e-10
            # Central finite differences of independently solved objective.
            direction=torch.randn_like(c);h=1e-5
            numerical=(loss(c.detach()+h*direction,k)-loss(c.detach()-h*direction,k))/(2*h)
            assert abs(float(numerical-(grad*direction).sum()))<1e-6
            cases.append(dict(type='centroid_gradient',batch=b,d=d,kappa=k,error=error))
    for b,d in [(3,6),(6,3),(4,4)]:
        for rank in range(1,min(b,d)+1):
            u=torch.randn(b,rank,dtype=torch.float64);u=torch.nn.functional.normalize(u,dim=1)
            u=torch.nn.functional.pad(u,(0,d-rank));v=u.clone()
            for k in [.1,1.,10.,100.]:
                ell=float(loss((u+v)/2,k));r=min(b,d)
                assert ell>=(r-rank)/(r+k*rank)-1e-10
                assert rank>=r*(1-ell)/(1+k*ell)-1e-10
                cases.append(dict(type='rank_certificate',batch=b,d=d,rank=rank,kappa=k,error=0.))
    for kind in ['constant','antipodal']:
        z=torch.randn(1,5,dtype=torch.float64).repeat(8,1).requires_grad_(True)
        zp=(z.detach().clone() if kind=='constant' else -z.detach().clone()).requires_grad_(True)
        u=torch.nn.functional.normalize(z,dim=1);v=torch.nn.functional.normalize(zp,dim=1)
        ell=loss((u+v)/2,10.);grad=torch.autograd.grad(ell,(z,zp))
        error=max(float(g.abs().max()) for g in grad);assert error<1e-12
        cases.append(dict(type='stationary_'+kind,error=error))
    return dict(cases=len(cases),maximum_residual=max(r['error'] for r in cases),records=cases)

if __name__ == "__main__":
    a=theory_checks(); b=additional_theory()
    print(json.dumps({"status":"passed","cases":a["cases"]+b["cases"],"maximum_residual":max(a["maximum_decomposition_error"],b["maximum_residual"])},indent=2))
