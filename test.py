import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
from free_guidance import StableDiffusionFreeGuidancePipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn import init
from utils.guidance_functions import *
import argparse
from diffusers import LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from utils import *
from PIL import Image
torch.cuda.manual_seed_all(1234) 
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
mpl.rcParams['image.cmap'] = 'gray_r'

print("complete")


print("Start Inference!")
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_id', type=str, default="/data/zsz/models/storage_file/models/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9")
# parser.add_argument('--seed', type=int, default=None)
# args = parser.parse_args()
# ded79e214aa69e42c24d3f5ac14b76d568679cc2
# model_id = "/data/zsz/models/storage_file/models/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
# model_id = "/data/zsz/models/storage_file/models/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionFreeGuidancePipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.unet = UNetDistributedDataParallel(pipe.unet, device_ids=[0]).cuda()
# pipe.vae = UNetDistributedDataParallel(pipe.vae, device_ids=[0,1,2]).cuda()
# pipe.text_encoder = UNetDistributedDataParallel(pipe.text_encoder, device_ids=[0,1,2]).cuda()
# pipe.unet = pipe.unet.to(device)
# pipe.text_encoder = UNetDistributedDataParallel(pipe.text_encoder, device_ids=[0,1,2,3,4], output_device=3).cuda()
# pipe.unet.config, pipe.unet.dtype, pipe.unet.attn_processors, pipe.unet.set_attn_processor = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.attn_processors, pipe.unet.module.set_attn_processor
# pipe.unet.config, pipe.unet.dtype = pipe.unet.module.config, pipe.unet.module.dtype
pipe.unet = pipe.unet.module
pipe = pipe.to(device)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
torch.backends.cudnn.benchmark = True

import os
for file in os.scandir("/home/nas4_user/sungwonhwang/ws_student/hyojinjang/2D/Free-Guidance-Diffusion/outputs"):
        os.remove(file.path)



print("non error")
seed = int(torch.rand((1,)) * 100000)
generator=torch.manual_seed(56572)
# generator=torch.manual_seed(21533)

print(seed)
prompt = 'a red balloon floating in the air'
object_to_edit = 'red balloon'
guidance = partial(edit_appearance_object, shape_weight=1)
image_list = pipe(prompt, obj_to_edit =object_to_edit, height=512, width=512, num_inference_steps=50, generator=generator,
        max_guidance_iter_per_step=1, guidance_func=guidance, g_weight=2500)
ls = ['edit10', 'ori10']
for i, image in enumerate(image_list):
    image.images[0].save(f"outputs/{seed}_{ls[i]}.png")
show_images([i for i in [image_list[0].images[0], image_list[1].images[0]]], titles=['edited', 'original'])

# ############################### 풍선 ##############################################################################################
# seed = int(torch.rand((1,)) * 100000)
# generator=torch.manual_seed(56572)
# # generator=torch.manual_seed(21533)

# print(seed)
# prompt = 'a red balloon floating in the air'
# object_to_edit = 'red balloon'
# move = partial(roll_shape, direction='down', factor=0.5)
# guidance = partial(move_object_by_shape, shape_weight=1, appearance_weight=1, position_weight=7, tau=move)
# image_list = pipe(prompt, obj_to_edit =object_to_edit, height=512, width=512, num_inference_steps=50, generator=generator,
#         max_guidance_iter_per_step=1, guidance_func=guidance, g_weight=2500)
# ls = ['edit10', 'ori10']
# for i, image in enumerate(image_list):
#     image.images[0].save(f"outputs/{seed}_{ls[i]}.png")
# show_images([i for i in [image_list[0].images[0], image_list[1].images[0]]], titles=['edited', 'original'])

############################### 풍선 ##############################################################################################


# seed = int(torch.rand((1,)) * 100000)
# generator=torch.manual_seed(21533)
# # generator=torch.manual_seed(56572)

# print(seed)
# prompt = 'a red balloon floating in the air'
# object_to_edit = 'red balloon'
# move = partial(roll_shape, direction='left', factor=1)
# guidance = partial(move_object_by_shape, shape_weight=1, appearance_weight=1, position_weight= 6, tau=move)
# image_list = pipe(prompt, obj_to_edit =object_to_edit, height=512, width=512, num_inference_steps=50, generator=generator,
#         max_guidance_iter_per_step=1, guidance_func=guidance, g_weight=1500)
# ls = ['edit', 'ori']
# for i, image in enumerate(image_list):
#     image.images[0].save(f"outputs/{seed}_{ls[i]}.png")
# show_images([i for i in [image_list[0].images[0], image_list[1].images[0]]], titles=['edited', 'original'])

# print("non error")


# seed = int(torch.rand((1,)) * 100000)
# generator=torch.manual_seed(77944)
# print(seed)
# prompt = 'a photo of an donut and a shot of espresso on a table'
# object_to_edit = 'donut'
# objects = ['donut', 'espresso']
# move = partial(roll_shape, direction='left', factor=0.3)
# guidance = partial(move_object_by_shape, shape_weight=1, appearance_weight=2, position_weight=6, tau=move)
# img_path = './img/donut.png'
# init_latents = get_latents_from_image(pipe, img_path, device)
# image_list = pipe(prompt, obj_to_edit = object_to_edit, height=512, width=512, 
#                   num_inference_steps=50, generator=generator, 
#         max_guidance_iter_per_step=1, guidance_func=guidance, g_weight=2500)
# # ls = ['edit', 'ori']
# # for i, image in enumerate(image_list):
# #     image.images[0].save(f"results/{seed}_{ls[i]}.png")
# show_images([i for i in [image_list[0].images[0], image_list[1].images[0]]], titles=['edited', 'original'], save_orig=True)


# Eq(13)
# seed = int(torch.rand((1,)) * 100000)
# generator=torch.manual_seed(seed)
# print(seed)
# prompt = 'a photo of a donut falling on a swimming pool'
# ori_prompt='a photo of a donut'
# objects = ['a donut']
# guidance = partial(edit_layout, appearance_weight=3)
# # 32*32
# feature_layer = pipe.unet.up_blocks[-2].resnets[-2]
# image_list = pipe(prompt, ori_prompt, objects=objects, height=512, width=512, num_inference_steps=35, generator=generator,
#         max_guidance_iter_per_step=5, guidance_func=guidance, g_weight=4500, feature_layer=feature_layer)
# # ls = ['edit', 'ori']
# # for i, image in enumerate(image_list):
# #     image.images[0].save(f"results/{seed}_{ls[i]}.png")

# show_images([i for i in [image_list[0].images[0], image_list[1].images[0]]], titles=['edited', 'original'])

# print("non error")

############################## resize ################################################


# seed = int(torch.rand((1,)) * 100000)
# generator=torch.manual_seed(seed)
# print(seed)
# prompt = 'a photo of a donut and a shot of coffee on a table'
# object_to_edit = 'donut'
# objects = ['donut', 'coffee']
# resize = partial(resize, scale_factor=0.5)
# guidance = partial(resize_object_by_shape, shape_weight=0.5, appearance_weight=0.5, size_weight=8, tau=resize)
# img_path = './img/coffee.png'
# init_latents = get_latents_from_image(pipe, img_path, device)
# image_list = pipe(prompt, obj_to_edit = object_to_edit, height=512, width=512,
#                   num_inference_steps=50, generator=generator, objects = objects, latents=init_latents,
#         max_guidance_iter_per_step=1, guidance_func=guidance, g_weight=500)

# ls = ['edit', 'ori']
# for i, image in enumerate(image_list):
#     image.images[0].save(f"outputs/{seed}_{ls[i]}.png")

# show_images([i for i in [image_list[0].images[0], image_list[1].images[0]]], titles=['edited', 'original'])

# print("non error")
