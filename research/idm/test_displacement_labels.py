import unittest
import numpy as np
import torch
from research.idm.extract_observed_displacements import labels_from_endpoints,SELECT
class LabelTests(unittest.TestCase):
 def test_exact_endpoint_difference_and_indices(self):
  states=np.arange(40,dtype=np.float32).reshape(2,20);valid=np.ones((2,15),bool)
  x=labels_from_endpoints(states,valid)
  torch.testing.assert_close(x['delta8'],torch.tensor(states[1,SELECT]-states[0,SELECT]))
  self.assertTrue(x['delta8_mask'].all());self.assertEqual(x['current_state20'].shape,(20,))
 def test_missing_not_filled_and_masks_distinct(self):
  states=np.arange(40,dtype=np.float32).reshape(2,20);valid=np.ones((2,15),bool)
  valid[1,1]=False;valid[0,14]=False;valid[0,5]=False
  x=labels_from_endpoints(states,valid)
  self.assertTrue(torch.isnan(x['delta8'][0]));self.assertTrue(torch.isnan(x['delta8'][7]))
  self.assertTrue(torch.isnan(x['endpoint_states20'][0,3:9]).all())
  self.assertTrue(x['structural_state_mask32'][:,:20].all());self.assertFalse(x['structural_state_mask32'][:,20:].any())
  self.assertFalse(x['endpoint_state_observed_mask'][1,0])
 def test_camera_mask_does_not_silently_become_state_mask(self):
  valid=np.ones((2,15),bool);valid[0,0]=False
  x=labels_from_endpoints(np.zeros((2,20),np.float32),valid)
  self.assertTrue(x['delta8_mask'].all());self.assertFalse(x['original_valid'][0,0])
if __name__=='__main__':unittest.main()
