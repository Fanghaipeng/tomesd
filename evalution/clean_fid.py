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
# image_folder = "/data/fanghaipeng/datasets/COCO2017/val2017_centercrop_1024"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of COCO2017 val is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-default"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of SD3 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome_save"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of ToMe 0.5 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.5-1.0_save"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of ToMe 0.5-1 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-tome-0.5-0.4-1.0"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of ToMe 0.4-1 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-ToMe-pruneTrue-0.5-0.5-0.5"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of ToMe 0.5 prune step 0 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-ToMe-pruneTrue-step30-0.5-0.5-0.5"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of ToMe 0.5 prune step 30 is: {score}")

# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-ToMe-pruneTrue-step0-0.5-0.5-1.0"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of ToMe 0.5-1 prune step 0 is: {score}")


# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/StableDiffusion3Pipeline-1024-1024-50-7.0-float16-ToMe-pruneFalse-mergexFalse-step20-0.5-0.5-1.0"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of ToMe 0.5-1 prune step 0 is: {score}")


# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/1024-1024-50-7.0-float16-ToMe_STD-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e0.5-metric1-unstep0,49-unlayer0,4,8,12,16,20,23"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of 1024-1024-50-7.0-float16-ToMe_STD-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e0.5-metric1-unstep0,49-unlayer0,4,8,12,16,20,23 is: {score}")


# image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/1024-1024-50-7.0-float16-ToMe-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e0.5-metric1-unstepNone-unlayerNone"
# score = fid.compute_fid(ref_folder, image_folder, mode="clean")
# print(f"The clean FID of /data1/fanghaipeng/project/sora/tomesd/SD3/samples/1024-1024-50-7.0-float16-ToMe-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e0.5-metric1-unstepNone-unlayerNone is: {score}")




image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/1024-1024-50-7.0-float16-ToMe_STD-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e1.0-metric4-unstepNone-unlayerNone"
score = fid.compute_fid(ref_folder, image_folder, mode="clean")
print(f"The clean FID of 1024-1024-50-7.0-float16-ToMe_STD-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e1.0-metric4-unstepNone-unlayerNone is: {score}")


image_folder = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples/1024-1024-50-7.0-float16-ToMe-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e1.0-metric1-unstepNone-unlayerNone"
score = fid.compute_fid(ref_folder, image_folder, mode="clean")
print(f"The clean FID of 1024-1024-50-7.0-float16-ToMe-pruneFalse-x:False-sa:True-ca:False-mlp:True-step0-s0.5-e1.0-metric1-unstepNone-unlayerNone is: {score}")
