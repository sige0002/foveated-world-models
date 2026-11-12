import unittest
import copy
import torch
from research.train_codec_predictor import CodecPredictor,evaluate,row_identity
from research.conditional_residual_codec import ConditionalResidualCodec


class CodecPredictionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        basis=torch.linalg.qr(torch.randn(16,4))[0]
        codec=ConditionalResidualCodec(basis,hidden=16,context_dim=8)
        self.model=CodecPredictor(codec,torch.zeros(4),torch.ones(4),code_dim=4,dim=16,layers=1,grid=3)
        self.h,self.b,self.y=[torch.randn(2,1,9,16) for _ in range(3)]
        self.z=torch.randn(2,1,4)

    def test_initial_direct_and_flow(self):
        latent=self.model.generate(self.h,self.z)
        torch.testing.assert_close(latent,torch.zeros_like(latent),rtol=0,atol=0)
        noise=torch.randn_like(latent)
        torch.testing.assert_close(self.model.generate(self.h,self.z,noise=noise),noise,rtol=0,atol=0)
        with self.assertRaises(TypeError):
            self.model.generate(self.h,self.z,future=self.y)

    def test_feature_gradients_keep_codec_frozen(self):
        latent=self.model.generate(self.h,self.z)
        loss=(self.model.decode(self.h,self.b,latent)-self.y).square().mean()
        loss.backward()
        self.assertGreater(self.model.output.weight.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in self.model.codec.parameters()))

    def test_evaluation_never_encodes_future(self):
        def forbidden(*args,**kwargs):
            raise AssertionError('Future encoder used during forecast evaluation')
        self.model.codec.encode=forbidden
        data={'current':self.h,'baseline':self.b,'future':self.y,'codes':self.z,
              'rows':[{'episode':str(i),'index':i,'endpoint_indices':[0,36],'dt':1.2} for i in range(2)]}
        for variant in ['direct_feature','flow','flow_low_noise']:
            result=evaluate(self.model,data,variant)
            self.assertTrue(torch.isfinite(torch.tensor(result['metrics']['mean_mse'])))

    def test_legacy_row_metadata(self):
        rows=[{'episode':str(i),'index':i,'dt':1.2} for i in range(2)]
        self.assertEqual(row_identity(rows),[('0',0,1.2,None),('1',1,1.2,None)])
        data={'current':self.h,'baseline':self.b,'future':self.y,'codes':self.z,'rows':rows}
        result=evaluate(self.model,data,'direct_feature')
        self.assertEqual(row_identity(result['rows']),row_identity(rows))

    def test_baseline_context_preserves_initial_weights_and_rng(self):
        codec=copy.deepcopy(self.model.codec)
        torch.manual_seed(73)
        base=CodecPredictor(codec,torch.zeros(4),torch.ones(4),code_dim=4,dim=16,layers=1,grid=3)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(73)
        model=CodecPredictor(copy.deepcopy(codec),torch.zeros(4),torch.ones(4),code_dim=4,dim=16,layers=1,grid=3,baseline_context=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        with torch.no_grad():
            base.output.weight.normal_(std=.1)
            model.output.weight.copy_(base.output.weight)
        torch.testing.assert_close(base.generate(self.h,self.z),model.generate(self.h,self.z,baseline=self.b),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            model.generate(self.h,self.z)
        model.generate(self.h,self.z,baseline=self.b).square().mean().backward()
        self.assertGreater(model.baseline_proj.weight.grad.norm().item(),0)


if __name__=='__main__':
    unittest.main()
