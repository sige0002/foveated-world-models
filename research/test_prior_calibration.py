import unittest
import torch
from research.calibrate_current_prior import fit,predict


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.b=torch.randn(17,5,7,dtype=torch.float64)
        self.h=torch.randn_like(self.b)

    def test_channel_affine(self):
        a=torch.linspace(.4,1.2,7);c=torch.linspace(-.2,.2,7)
        y=self.b*a+c;m=fit(self.b,self.h,y)
        torch.testing.assert_close(predict('channel_affine',m['channel_affine'],self.b,self.h),y,atol=1e-12,rtol=1e-12)

    def test_patch_mix(self):
        a=torch.arange(5,dtype=torch.float64)[:,None]/5;c=1-a
        y=a*self.b+c*self.h;m=fit(self.b,self.h,y)
        torch.testing.assert_close(predict('patch_mix',m['patch_mix'],self.b,self.h),y,atol=1e-12,rtol=1e-12)

    def test_degenerate_prior(self):
        b=torch.zeros_like(self.b);y=self.h+2
        m=fit(b,self.h,y)
        pred=predict('channel_affine',m['channel_affine'],b,self.h)
        torch.testing.assert_close(pred.expand_as(y),y.mean((0,1)).expand_as(y))


if __name__=='__main__':unittest.main()
