import os
if 'PCam_context' not in os.getcwd():
    os.chdir('PCam_context')
import torch
from vit import VisionTransformer
from vit_moco import VisionTransformerMoCo
from vit_mae import VisionTransformer as VisionTransformerMAE
from functools import partial
from swin_transformer import SwinTransformer 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = "./pretrained_models/"
SWIN = PATH+"pcamswin_4_0.915008544921875_full_no_standarization.pth"
MAE = PATH+"pcammae_vitb16_3_0.912261962890625_full_no_standarization.pth"
MOCO = PATH+"pcammocov3_vitb_3_0.91387939453125_full_no_standarization.pth"
SUP = PATH+"pcamsup_vitb16_imagenet21k_1_0.9102783203125_full_no_standarization.pth"
checkpoints = {'pcamswin':SWIN, 'pcammae':MAE, 'pcammoco':MOCO, 'pcamsup':SUP}

def build_swin_model():
    model = SwinTransformer(
        img_size=(224, 224),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
        num_classes=-1,
    )
    embed_dim = 128
    num_layers = 4
    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    model.head=torch.nn.Linear(1024, 2)
    return model

def build_model(name):
    model = None
    if name == 'pcamswin':
        model = build_swin_model()
    elif name == 'pcammae':
        model = VisionTransformerMAE(
            drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
        )
        model.head=torch.nn.Linear(768,2)
    elif 'pcammoco' in name:
        model = VisionTransformerMoCo(patch_size=16, embed_dim=768, depth=12, 
            num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
        model.head=torch.nn.Linear(768,2)
    elif name == 'pcamsup':
        model = VisionTransformer(model_type="sup_vitb16_imagenet21k", num_classes=2)
    return model

def get_transformer_model(name, device='cpu'):
    model = build_model(name)
    if 'pcammoco' in name:
        name = 'pcammoco'
    ckpt = checkpoints[name]  
    state_dict = torch.load(ckpt, map_location=device)
    for k in list(state_dict.keys()):
        if k.startswith('enc.'):
            state_dict[k[len("enc."):]] = state_dict[k]
            del state_dict[k]
        if k.startswith('head.'):                                       
            state_dict[k.replace('last_layer.', '')] = state_dict[k]
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model
