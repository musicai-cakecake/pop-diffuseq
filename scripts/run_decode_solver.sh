CUDA_VISIBLE_DEVICES=1 python -u run_decode_solver.py \
--model_dir diffusion_models/diffuseq_lmd_matched_h128_lr0.0001_t2000_sqrt_lossaware_seed102_MusicDiffuseq-ACMMM-lmd_matched-240320240407-13:45:59/ \
--seed 110 \
--bsz 40 \
--step 10 \
--split test
