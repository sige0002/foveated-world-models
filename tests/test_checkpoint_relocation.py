from train_stage1 import relocated_checkpoint_state


def test_alias_preserves_best_checkpoint_metadata(tmp_path):
    target=tmp_path/'ssd1';target.mkdir()
    old=tmp_path/'old';old.symlink_to(target,target_is_directory=True)
    state={'dirpath':str(old),'best_model_path':str(old/'best.ckpt'),
           'last_model_path':str(old/'last.ckpt'),'best_k_models':{str(old/'best.ckpt'):.25},
           'best_model_score':.25}
    actual=relocated_checkpoint_state(state,str(target))
    assert actual['dirpath']==str(target)
    assert actual['best_k_models']=={str(target/'best.ckpt'):.25}
    assert actual['best_model_score']==.25
    assert state['dirpath']==str(old)


def test_unrelated_directory_not_rewritten(tmp_path):
    state={'dirpath':str(tmp_path/'other'),'best_k_models':{}}
    assert relocated_checkpoint_state(state,tmp_path/'new') is state
