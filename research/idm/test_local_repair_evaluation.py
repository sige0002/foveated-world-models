"""Synthetic evaluator checks; no real checkpoint/probe fitting is performed."""
import tempfile,json,unittest
from pathlib import Path
import torch
from research.idm.local_repair_evaluation_common import validate_run_map,gaussian_pair_metrics,check_completed,load_completed_model
from research.idm.evaluate_local_repair_probes import select_train_scores,CONFIGS,ALPHAS
from research.idm.probe_interaction512 import combine
from research.idm.train_local_repair import ROOT,RepairModel,anchors,state_hash,tensor_hash
from research.idm.test_local_repair import TinyDecoder
from torch import nn
from research.train_cached_decoders import file_hash

class EvaluationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_all_runs_required_no_observed_choice(self):
        mapping={f'{a}{s}':f'/not-loaded/{a}{s}' for a in ('H','D','HD') for s in (42,43,44)}
        self.assertEqual(len(validate_run_map(mapping)),9)
        del mapping['H43']
        with self.assertRaises(ValueError):validate_run_map(mapping)
    def test_gaussian_metrics_use_each_fixed_variance(self):
        mu0=torch.zeros(2,1,3,dtype=torch.float64);mu1=torch.ones_like(mu0)*.4
        lv0=torch.zeros_like(mu0);lv1=torch.ones_like(mu0)*.2
        metrics=gaussian_pair_metrics(mu0,mu1,lv0,lv1)
        a=torch.distributions.Normal(mu0,(.5*lv0).exp());b=torch.distributions.Normal(mu1,(.5*lv1).exp())
        exact=(torch.distributions.kl_divergence(a,b)+torch.distributions.kl_divergence(b,a)).flatten(1).sum(1)/2
        torch.testing.assert_close(metrics['symmetric_kl'],exact)
        same=gaussian_pair_metrics(mu0,mu0,lv0,lv0)
        self.assertTrue(all(torch.equal(v,torch.zeros_like(v)) for v in same.values()))
        same_mean=gaussian_pair_metrics(mu0,mu0,lv0,lv1)
        self.assertTrue((same_mean['symmetric_kl']>0).all())
    def test_train_selection_is_all_candidates_and_train_only(self):
        cv={'F':{c:torch.ones(len(ALPHAS),6,dtype=torch.float64) for c in CONFIGS}}
        cv['F']['spatial_rbf1'][6]=.2
        result=select_train_scores(cv)['F']
        self.assertEqual(result['selected']['config'],'spatial_rbf1')
        self.assertEqual(result['selected']['alpha'],ALPHAS[6])
        self.assertEqual(len(result['candidates']),len(CONFIGS)*len(ALPHAS))
    def test_frozen_code_alias_same_kernel(self):
        g=torch.Generator().manual_seed(8);x=torch.randn(12,5,generator=g,dtype=torch.float64);z=torch.randn(12,32,generator=g,dtype=torch.float64)
        gram=x@x.T/5
        k,q,_=combine(gram,gram,gram.diag(),z,z,1,'interaction')
        kd,qd,_=combine(gram,gram,gram.diag(),z.clone(),z.clone(),1,'interaction')
        self.assertTrue(torch.equal(k,kd));self.assertTrue(torch.equal(q,qd))
    def test_completed_mean_only_checkpoint_roundtrip(self):
        torch.manual_seed(51);head=nn.Sequential(nn.Linear(5,7),nn.GELU(),nn.Linear(7,64));decoder=TinyDecoder()
        x=torch.randn(4,1,5);reference=anchors(head,{'train':x});constant=reference['train']['mu'].mean(0,keepdim=True)
        model=RepairModel(head,decoder,constant,'D');ident=dict(head=state_hash(model.head),decoder=state_hash(model.decoder),constant_mu=tensor_hash(model.constant_mu))
        with tempfile.TemporaryDirectory(dir=ROOT/'research/idm') as tmp:
            path=Path(tmp);protocol=dict(arm='D',minibatch_seed=42,steps=400,execute_training=True,sources={})
            (path/'protocol.json').write_text(json.dumps(protocol));(path/'preflight.json').write_text('{}')
            ck=dict(arm='D',minibatch_seed=42,steps=400,source_hashes={},initial=ident,final=ident,state_dict=model.state_dict(),reference_anchor_hashes={s:{k:tensor_hash(v) for k,v in d.items()} for s,d in reference.items()})
            torch.save(ck,path/'repair.pt')
            result=dict(checkpoint_sha256=file_hash(path/'repair.pt'),protocol_sha256=file_hash(path/'protocol.json'),preflight_sha256=file_hash(path/'preflight.json'),frozen_audit_passed=True,final=ident,drift_scope_failed=False)
            (path/'results.json').write_text(json.dumps(result))
            loaded,_=load_completed_model(path,'D',42,head,decoder,constant,{},reference)
            self.assertTrue(torch.equal(loaded.mean(x),model.mean(x)))
            self.assertEqual(loaded.head[-1].out_features,32)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_incomplete_and_wrong_endpoint_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'research/idm') as tmp:
            path=Path(tmp)
            with self.assertRaises(FileNotFoundError):check_completed(path,'H',42,{})
            protocol=dict(arm='H',minibatch_seed=42,steps=399,execute_training=True,sources={})
            (path/'protocol.json').write_text(json.dumps(protocol));(path/'preflight.json').write_text('{}')
            torch.save({},path/'repair.pt')
            result=dict(checkpoint_sha256=file_hash(path/'repair.pt'),protocol_sha256=file_hash(path/'protocol.json'),preflight_sha256=file_hash(path/'preflight.json'),frozen_audit_passed=True)
            (path/'results.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):check_completed(path,'H',42,{})

if __name__=='__main__':unittest.main()
