"""Synthetic gradient-route checks only; no dataset training or GPU use."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from research.idm.train_joint_objectives import JointModel,clip_separately

class GradientRoutes(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);torch.manual_seed(42)
        self.model=JointModel(query_dim=6,code_dim=4,feature_dim=8,dim=16,layers=1,grid=2)
        # Nonzero gates test routing after initialization; zero AdaLN gates legitimately block z at step zero.
        for branch in (self.model.decoder,self.model.flow):
            for block in branch.blocks:torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.05)
        self.args=(torch.randn(2,1,6),torch.randn(2,1,4,8),torch.randn(2,1,4,8),torch.randn(2,1,4),torch.randn(2,1,4,8),torch.tensor([.2,.8]))
    def flatgrad(self,loss,parameters):
        ps=list(parameters);gs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(ps,gs)])
    def test_detach_preserves_losses_and_flow_gradients(self):
        b=copy.deepcopy(self.model);c=copy.deepcopy(self.model)
        lb=b.losses(*self.args,'B');lc=c.losses(*self.args,'C')
        for key in lb:self.assertTrue(torch.equal(lb[key],lc[key]))
        gb=self.flatgrad(lb['flow'],b.head.parameters());gc=self.flatgrad(lc['flow'],c.head.parameters())
        self.assertGreater(gb.norm().item(),0);self.assertEqual(gc.norm().item(),0)
        self.assertTrue(torch.equal(self.flatgrad(lb['flow'],b.flow.parameters()),self.flatgrad(lc['flow'],c.flow.parameters())))
        self.assertTrue(torch.equal(self.flatgrad(lb['direct'],b.head.parameters()),self.flatgrad(lc['direct'],c.head.parameters())))
    def test_gradient_difference_is_exact_flow_contribution(self):
        b=copy.deepcopy(self.model);c=copy.deepcopy(self.model)
        lb=b.losses(*self.args,'B');lc=c.losses(*self.args,'C');weight=.7
        gb=self.flatgrad(lb['direct']+5e-5*lb['kl']+weight*lb['flow'],b.head.parameters())
        gc=self.flatgrad(lc['direct']+5e-5*lc['kl']+weight*lc['flow'],c.head.parameters())
        expected=weight*self.flatgrad(lb['flow'],b.head.parameters())
        self.assertTrue(torch.allclose(gb-gc,expected,atol=1e-7,rtol=1e-5))
    def test_a_c_same_head_direct_updates(self):
        a=copy.deepcopy(self.model);c=copy.deepcopy(self.model)
        opts=[torch.optim.AdamW(m.parameters(),lr=1e-3) for m in (a,c)]
        for _ in range(3):
            for model,opt,arm in zip((a,c),opts,('A','C')):
                p=model.losses(*self.args,arm);loss=p['direct']+5e-5*p['kl']
                if p['flow'] is not None:loss=loss+p['flow']
                opt.zero_grad(set_to_none=True);loss.backward();clip_separately(model);opt.step()
        for name in ('head','decoder'):
            for left,right in zip(getattr(a,name).parameters(),getattr(c,name).parameters()):self.assertTrue(torch.equal(left,right))
    def test_narrow_and_wide_factories(self):
        for family in ('narrow','wide'):
            model=JointModel(query_dim=6,code_dim=4,feature_dim=8,dim=16,layers=1,grid=2,decoder_family=family)
            for block in model.flow.blocks:torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.05)
            parts=model.losses(*self.args,'B')
            self.assertGreater(self.flatgrad(parts['flow'],model.head.parameters()).norm().item(),0)
            self.assertEqual(model.decoder_family,family)

    def test_interpolation_is_future_path(self):
        seen={}
        def velocity(x,t,current,actions):seen['x']=x.detach();return torch.zeros_like(x)
        self.model.flow.velocity=velocity
        q,current,target,eps,noise,time=self.args
        self.model.losses(*self.args,'B')
        t=time.reshape(-1,1,1,1);expected=(1-t)*(current+.1*noise)+t*target
        self.assertTrue(torch.equal(seen['x'],expected))
        with self.assertRaises(TypeError):self.model.flow(current,eps,noise=noise,target=target)

if __name__=='__main__':unittest.main()
