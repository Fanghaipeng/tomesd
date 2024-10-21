import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from pipeline_stable_diffusion_3_SaveStepOutput_TimeCount import StableDiffusion3Pipeline
from tome import apply_patch_tome
def create_save_dir(text, ratio_start, ratio_end):
    # 分词
    words = text.split()
    
    if len(words) < 5:
        selected_words = words
    else:
        selected_words = words[-5:]
    
    # 将选定的词与比率参数连接成一个字符串，使用下划线作为分隔符
    save_dir = f"{ratio_start}_{ratio_end}_" + "_".join(selected_words)
    return save_dir

pipe = StableDiffusion3Pipeline.from_pretrained("/data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
generator = torch.Generator(device="cuda").manual_seed(30)
pipe = pipe.to("cuda")
height = 1024
width = 1024
text = "a photo of an astronaut riding a horse on mars"
ratio_start = 0.5
ratio_end = 0.5
save_dir = create_save_dir(text, ratio_start, ratio_end)
save_dir = os.path.join("/data1/fanghaipeng/project/sora/tomesd/SD3/test/tome", save_dir)
print(save_dir)
apply_patch_tome(pipe, ratio=0.5, ratio_start=ratio_start, ratio_end=ratio_end, sx=2, sy=2, save_dir=save_dir) # Can also use pipe.unet in place of pipe here
image = pipe(
	"a photo of an astronaut riding a horse on mars",
	generator=generator,
    height=height,
    width=width,
    num_inference_steps=50,
    guidance_scale=7.0,
).images[0]
# image.save("astronaut_50_tome_sd3_0.5.png")