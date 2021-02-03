python main.py --scale 2 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x2/model/model_best.pt --test_only --model EDSR --data_range '1-800/801-900' --dir_data /tigress/zejiangh/unknown_downsampling


python main.py --model EDSR --scale 2 --reset --check_prune --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/pruned_model_edsr_baseline_x2_0.3/best_ckpt.pth.tar --data_test DIV2K --prune_prob 0.3 --n_feats 44 --data_range '1-800/801-900' --self_ensemble --dir_data /tigress/zejiangh/unknown_downsampling

python main.py --model EDSR --scale 4 --reset --check_prune --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/pruned_model_edsr_baseline_x4_interpolate_0.72/best_ckpt.pth.tar --prune_prob 0.72 --n_feats 17 --data_range '1-800/801-900' --self_ensemble
--dir_data /tigress/zejiangh/unknown_downsampling


python main.py --model EDSR --scale 4 --reset --flickr_finetune --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/pruned_model_edsr_baseline_x4_interpolate_0.72/best_ckpt.pth.tar --prune_prob 0.72 --n_feats 17 --epoch 400 --dir_data /tigress/zejiangh/Flickr2K/ --data_range '1-2600/2601-2650' --save tmp --decay '50-100-200-300' --n_GPUs 2



python main.py --model EDSR --scale 4 --patch_size 192 --save edsr_baseline_x4_unknown --reset --epochs 200 --n_GPUs 2 --data_range '1-800/801-900' --lr 1e-4 --dir_data /tigress/zejiangh/unknown_downsampling --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4/model/model_best.pt

python main.py --model EDSR --scale 4 --patch_size 192 --reset --check_prune --temp_save ? --prune_prob 0.7 --n_feats 19 --data_range '1-800/801-900' --dir_data /tigress/zejiangh/unknown_downsampling --epochs 300 --n_GPUs 2 --save pruned_edsr_x4_0.7_unknown_Q100 --data_train DIV2K-Q100 --data_test DIV2K-Q100


python main.py --model HSARN --scale 4 --reset --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/HSARN_scalex4_epoch200to400/model/model_best.pt --n_resblocks 10 --n_resgroups 10 --data_test Urban100  --test_only --self_ensemble


python main.py --model STSR --scale 4 --patch_size 192 --reset --save stsr_x4 --epoch 1 --data_range '1-800/801-900' --trainer_v 3

python main.py --model EDSR --scale 2 --patch_size 96 --reset --data_test Set5+Set14+B100 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x2_r44/model/model_best.pt --prune --temp_save ./pruned_model_edsr_baseline_x2_r44 --save ./pruned_edsr_baseline_x2_r44 --n_feats 44




python main.py --data_test Urban100 --scale 4 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4_interpolate/model/model_best.pt --test_only --model EDSR --save_results --chop

python main.py --model EDSR --scale 4 --patch_size 192  --save ERF_layer1 --reset --epochs 1000 --cpu --n_resblocks 1 --data_range '1-800/801-900'


python main.py --model EDSR --scale 4 --patch_size 256 --reset --save edsr_baseline_x2_r32_Lin --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4_r32/model/model_best.pt --n_feats 32 --n_GPUs 2

python main.py --data_test B100 --scale 4 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4/model/model_best.pt --test_only --n_feats 64 --save_results --self_ensemble

python main.py --template RCAN --save RCAN_x2 --scale 2 --reset --patch_size 96 --data_range 801-810 --n_GPUs 2

python main.py --model RRDB --save RRDB_x2 --scale 2 --reset --patch_size 96 --lr 2e-4 

python main.py --model EDSR --scale 4 --patch_size 192 --reset --save edsr_baseline_x4_r48_cont --n_feats 48 --n_GPUs 1 --epochs 400 --data_range '1-800/801-900' --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4_r48/model/model_best.pt --lr 1.25e-5

python main.py --model EDSR --scale 4 --patch_size 192 --reset --save temp --epochs 1 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4/model/model_best.pt

python main.py --model EDSR --scale 2 --patch_size 96 --reset --save edsr_baseline_x2_r48_cont --n_feats 48 --epochs 200 --data_range '1-800/801-900' --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x2_r48/model/model_best.pt --lr 2.5e-5

python main.py --model EDSR --scale 1 --patch_size 48 --save edsr_baseline_x1 --reset --data_train DIV2K-Q1 --data_test DIV2K-Q1 --data_range '1-800/801-900' --epochs 1000

python main.py --scale 2 --save RDN_x2_q75 --model RDN --epochs 1000 --decay '200-400-600-800' --batch_size 16 --data_range '1-800/801-810' --patch_size 96 --reset --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/RDN_x2_Lin_cont/model/model_best.pt --n_GPUs 4 --data_train DIV2K-Q75 --data_test DIV2K-Q75

python main.py --data_test Demo --scale 2 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x2/model/model_best.pt --test_only --save_results --model EDSR

python main.py --model EDSRMS --scale 4 --patch_size 192 --save edsrms_G16_B4_x4_cont --reset --epochs 800 --n_GPUs 3 --data_range '1-800/801-810' --n_feats 64 --n_resblocks 16 --n_convs 4 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsrms_G16_B4_x4/model/model_best.pt --decay '200-400-600' --lr 5e-5

python main.py --data_test Urban100 --scale 2 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/rcan+_G20_B6_x2_cont/model/model_best.pt --test_only --model RCANPLUS --n_resblocks 20 --n_convs 6 --self_ensemble

python main.py --model EDSR --scale 1 --reset --n_feats 19 --data_test Set14 --test_only --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x1_C30_r19/model/model_best.pt --save_results

python main.py --model EDSR --scale 4 --save edsr_baseline_x4_r32 --reset --n_feats 32 --epochs 200 --data_range '1-800/801-900' --n_GPUs 2

python main.py --model EDSR --scale 4 --patch_size 192 --reset --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4/model/model_best.pt --prune --temp_save ./pruned_model_edsr_baseline_x4_0.5_unsupervised --save ./pruned_edsr_baseline_x4_0.5_unsupervised --prune_prob 0.5 --epochs 200 --data_range '1-800/801-900' --unsupervised_pruning --n_GPUs 1

python main.py --model EDSR --scale 4 --patch_size 192 --reset --check_prune --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/src/pruned_model_edsr_baseline_x4_0.3_unsupervised/best_ckpt.pth.tar --data_test Set5+Set14+B100+Urban100 --prune_prob 0.3 --n_feats 44

python main.py --model EDSR --scale 4 --patch_size 192 --reset --check_prune --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/src/pruned_model_edsr_baseline_x4_0.5/best_ckpt.pth.tar --data_test Set5+Set14+B100+Urban100 --prune_prob 0.5 --n_feats 32 --self_ensemble

python main.py --model EDSR --scale 4 --patch_size 192 --save edsr_gan_x4_0.5 --reset --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/src/pruned_model_edsr_baseline_x4_0.5/best_ckpt.pth.tar --prune_prob 0.5 --n_feats 32 --data_range '1-800/801-900' --n_GPUs 2  --loss 5*VGG54+0.15*GAN --pruned_for_gan --template GAN

python main.py --model EDSR --scale 4 --patch_size 192 --save edsr_baseline_x4_se --reset --epoch --data_range '1-800/801-900' --n_GPUs 2 --epochs 300


python main.py --model EDSR --scale 2 --patch_size 96 --data_train DIV2K-Q75 --data_test DIV2K-Q75 --data_range '1-800/801-900' --save pruned_edsr_baseline_x2_0.5_Q75 --reset --prune_jpeg --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/pruned_model_edsr_baseline_x2_0.5/best_ckpt.pth.tar --n_feats 32 --epoch 1000 --decay '200-400-600-800' --n_GPUs 2


python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --data_range '1-800/801-810' --n_GPUs 2

python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_mobile_x2 --reset --eopchs 1000 --n_GPUs 2 --data_range '1-800/801-900'

python main.py --model EDSR --scale 4 --patch_size 192 --save edsr_baseline_x4_interpolate_G16_k1 --reset --epochs 1000 --n_GPUs 2 --data_range '1-800/801-900' --n_resblocks 16 --n_feats 64 

python main.py --model EDSR --scale 4 --patch_size 192 --save edsr_baseline_x4_unknown --reset --epochs 200 --n_GPUs 2 --data_range '1-800/801-900' --lr 1e-4 --dir_data /tigress/zejiangh/unknown_downsampling --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4/model/model_best.pt

python main.py --model EDSR --scale 4 --patch_size 192 --reset --check_prune --temp_save /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/pruned_model_edsr_baseline_x4_0.7/best_ckpt.pth.tar --prune_prob 0.7 --n_feats 19 --data_range '1-800/801-900' --dir_data /tigress/zejiangh/unknown_downsampling --epochs 1 --n_GPUs 1 --save pruned_edsr_x4_0.7_unknown

python main.py --data_test Set5+Set14+B100+Urban100 --scale 4 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/srresnet_CA_epoch200/model/model_best.pt --test_only --model EDSR --n_feats 64 --n_resblocks 20 --self_ensemble

## RCANPLUS experiments

python main.py --model RCANPLUS --scale 4 --patch_size 192 --save rcan+_exp2_epoch200 --reset --epochs 200 --n_GPUs 3 --data_range '1-800/801-810' --n_resblocks 20 --n_convs 5 --lr 1e-4

python main.py --data_test Set5+Set14+B100 --scale 4 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/rcan+_exp2_epoch200/model/model_best.pt --test_only --self_ensemble --model RCANPLUS --data_range '1-800/801-900' --n_resblocks 20 --n_convs 5


python main.py --model HSARN --scale 4 --reset --n_resblocks 10 --n_resgroups 10 --data_test Set5+Set14+B100+Urban100 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/HSARN_scalex4/model/model_best.pt

python main.py --model EDSR --data_test DIV2K --data_range 801-900 --scale 2 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x2/model/model_best.pt --test_only


python main.py --model EDSR --data_test DIV2K --data_range 801-810 --scale 4 --pre_train /home/zejiangh/l1-norm-pruning/EDSR-PyTorch-legacy-0.4.1/experiment/edsr_baseline_x4/model/model_best.pt --test_only --n_resblocks 16






