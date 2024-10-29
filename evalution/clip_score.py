import json
import os
from PIL import Image
import torch
from tqdm import tqdm
import open_clip
def calculate_mean_clip_score(json_path, image_folder):
    # 载入模型和处理器
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    # 载入 JSON 文件
    with open(json_path, "r") as file:
        captions = json.load(file)

    # 准备存储分数
    clip_scores = []

    # 遍历字典计算 CLIP 分数
    for item in tqdm(captions):
        image_id = str(item["image_id"])
        caption = item["caption"]

        # 尝试直接格式和带前导0的格式
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_folder, f"{image_id.zfill(12)}.jpg")

        # 载入图像
        if os.path.exists(image_path):
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            text = tokenizer([caption])
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (image_features @ text_features.T)
                # print(text_probs)

            # 保存 CLIP 分数
            clip_scores.append(text_probs[0].item())
        else:
            # print(f"Image not found for ID: {image_id}")
            pass

    # 计算平均 CLIP 分数
    mean_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0
    return mean_clip_score

# 使用函数
json_path = "/data/fanghaipeng/datasets/COCO2017/longest_captions.json"
image_folder = "/data/fanghaipeng/datasets/COCO2017/val2017"
# mean_score = calculate_mean_clip_score(json_path, image_folder)
# print(f"The mean CLIP score of COCO2017 val is: {mean_score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-default"
# mean_score = calculate_mean_clip_score(json_path, image_folder)
# print(f"The mean CLIP score of SD3 is: {mean_score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome_save"
# mean_score = calculate_mean_clip_score(json_path, image_folder)
# print(f"The mean CLIP score of ToMe 0.5 is: {mean_score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.5-1.0"
# mean_score = calculate_mean_clip_score(json_path, image_folder)
# print(f"The mean CLIP score of ToMe 0.5-1 is: {mean_score}")

image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.4-1.0"
mean_score = calculate_mean_clip_score(json_path, image_folder)
print(f"The mean CLIP score of ToMe 0.4-1 is: {mean_score}")