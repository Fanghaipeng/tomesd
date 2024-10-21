# from cleanfid import fid
# ref_folder = "/data/fanghaipeng/datasets/COCO2017/val2017_centercrop_1024"
# image_folder = "/data/fanghaipeng/datasets/COCO2017/val2017_centercrop_1024"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean", model_name="clip_vit_b_32")
# print(f"The clean FID of COCO2017 val is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-default"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean", model_name="clip_vit_b_32")
# print(f"The clean FID of SD3 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome_save"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean", model_name="clip_vit_b_32")
# print(f"The clean FID of ToMe 0.5 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.5-1.0"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean", model_name="clip_vit_b_32")
# print(f"The clean FID of ToMe 0.5-1 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.4-1.0"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean", model_name="clip_vit_b_32")
# print(f"The clean FID of ToMe 0.4-1 is: {score}")



from cleanfid import fid
ref_folder = "/data/fanghaipeng/datasets/COCO2017/val2017_centercrop_1024"
image_folder = "/data/fanghaipeng/datasets/COCO2017/val2017_centercrop_1024"
score = fid.compute_fid(ref_folder, image_folder, mode="clean")
print(f"The clean FID of COCO2017 val is: {score}")

image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-default"
score = fid.compute_fid(ref_folder, image_folder, mode="clean")
print(f"The clean FID of SD3 is: {score}")

image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome_save"
score = fid.compute_fid(ref_folder, image_folder, mode="clean")
print(f"The clean FID of ToMe 0.5 is: {score}")

image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.5-1.0"
score = fid.compute_fid(ref_folder, image_folder, mode="clean")
print(f"The clean FID of ToMe 0.5-1 is: {score}")

image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.4-1.0"
score = fid.compute_fid(ref_folder, image_folder, mode="clean")
print(f"The clean FID of ToMe 0.4-1 is: {score}")