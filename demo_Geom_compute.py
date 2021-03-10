#%%
# Define the image distance metric using LPIPS. Use squeezenet to save memory.
from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN, upconvGAN
import torch
import lpips
ImDist = lpips.LPIPS(net="squeeze", )

#%%
BGAN = loadBigGAN()  # Default to be "biggan-deep-256"
BGAN.cuda().eval()
BGAN.requires_grad_(False)
G = BigGAN_wrapper(BGAN)
#%%
feat = G.sample_vector(device="cuda", class_id=321).detach().clone()
eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=10)

#%%
# eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=30)
# eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
#%%
from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve
mtg, codes_all, = vis_eigen_explore(feat.cpu().numpy(), evc_FI, eva_FI, G, ImDist=None, eiglist=[1,2,3,4,7], transpose=False,
      maxdist=0.6, scaling=None, rown=7, sphere=False, distrown=15,
      save=False, namestr="demo")
#%%
vis_distance_curve(feat.cpu().numpy(), evc_FI, eva_FI, G, ImDist, eiglist=[1,2,3,4,7],
                   maxdist=0.6, rown=7, sphere=False, distrown=15, namestr="demo")

#%% DCGAN
# DG = loadDCGAN()
# DG.cuda().eval()
# DG.requires_grad_(False)
# G = DCGAN_wrapper(DG)
# feat = torch.randn(1, 120).detach().clone().cuda()
# # feat = G.sample_vector().detach().clone()
# eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=60)
# eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=60)
# eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
#
# #%% BigGAN
# BGAN = loadBigGAN()  # Default to be "biggan-deep-256"
# BGAN.cuda().eval()
# BGAN.requires_grad_(False)
# G = BigGAN_wrapper(BGAN)
# feat = G.sample_vector().detach().clone()  #0.05 * torch.randn(1, 256).detach().clone().cuda()
# eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=30)
# eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=30)
# eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")