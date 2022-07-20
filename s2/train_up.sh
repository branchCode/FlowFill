python3 train_up.py --name celebahq_up --dataset_mode inpaint --dataroot datasets/flist \
            --dataset_name celeba_hq --mask_type 4 --pconv_level 0 --load_size 256 --niter 30 --niter_decay 30 \
            --netG Upsampler --batchSize 4 --ngf 64 --n_layers_D 4 --input_nc 4 --output_nc 3 \
            --gan_mode hinge --use_attention --lambda_perceptual 0.001 --vgg_normal_correct \
            --lambda_hole 6  --lambda_vgg 2 --lambda_style 0 --lr 2e-4 \
            --gpu_ids 3,4 --continue_train \
            # --upsampler --sample_path ./samples 