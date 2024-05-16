"""
Contains a function for predict single image with selected model and parameters
"""
from torch import nn 
import torch 
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from typing import Dict,List,Tuple
import torchvision


device="cuda" if torch.cuda.is_available() else "gpu"


def pred_and_plot_image(
    model:torch.nn.Module,
    image_path:str,
    class_names:List[str]=None,
    transform=None,
    device=device):
    
    target_image=torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    target_image =target_image /255 
    
    if transform:
        target_image=transform(target_image)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        target_image=target_image.unsqueeze(0)
        
        target_image_pred=model(target_image.to(device))
        
        target_image_pred_probs=torch.softmax(target_image_pred,dim=1)
        
        target_image_pred_labels=torch.argmax(target_image_pred_probs,dim=1)
        
        plt.imshow(target_image.squeeze().permute(1,2,0))
        if class_names:
            title = f"Pred: {class_names[target_image_pred_labels.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        else:
            title = f"Pred: {target_image_pred_labels} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        plt.title(title)
        plt.axis(False)
