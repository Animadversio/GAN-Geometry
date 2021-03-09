from core.GAN_hessian_compute import get_full_hessian, hessian_compute
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN
import torch
import lpips
#%% DCGAN
ImDist = lpips.LPIPS(net="squeeze", )
DG = loadDCGAN()
DG.cuda().eval()
DG.requires_grad_(False)
G = DCGAN_wrapper(DG)
feat = torch.randn(1, 120).detach().clone().cuda()
# feat = G.sample_vector().detach().clone()
eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=60)
eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=60)
eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")

#%%
BGAN = loadBigGAN()  # Default to be "biggan-deep-256"
BGAN.cuda().eval()
BGAN.requires_grad_(False)
G = BigGAN_wrapper(BGAN)
feat = G.sample_vector().detach().clone()  #0.05 * torch.randn(1, 256).detach().clone().cuda()
eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=128)
eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=128)
eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")