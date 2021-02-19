from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import torch
def show_imgrid(img_tsr, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid(img_tsr.cpu(), *args, **kwargs))
    PILimg.show()
    return PILimg

def save_imgrid(img_tsr, path, *args, **kwargs):
    PILimg = ToPILImage()(make_grid(img_tsr.cpu(), *args, **kwargs))
    PILimg.save(path)
    return PILimg
