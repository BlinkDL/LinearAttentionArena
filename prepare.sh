#!/bin/bash
#######################################################################################################################
#
# !!! RUN THIS FIRST !!! This will generate the initial model, and save it to the output folder
#
#######################################################################################################################
#
# Please firstly create data folder & Download minipile (1498226207 tokens, around 3GB)
# mkdir -p data
# wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
# wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
#
#######################################################################################################################

model_type=""
layer=0
emb=0
ctx_len=512 # !!! change magic_prime if you change ctx_len !!!
suffix=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_type) model_type="$2"; shift ;;
        --layer) layer="$2"; shift ;;
        --emb) emb="$2"; shift ;;
        --ctx_len) ctx_len="$2"; shift ;;
        --suffix) suffix="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "model_type: $model_type"
echo "layer: $layer"
echo "emb: $emb"
echo "ctx_len: $ctx_len"
echo "suffix: $suffix"

PROJ_DIR="out/L"$layer"-D"$emb"-"$model_type$suffix # set output folder
echo "Saving to "$PROJ_DIR
rm -f "$PROJ_DIR"/rwkv-{0..100}.pth # remove old checkpts in folder

#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
python train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "data/minipile" --data_type "binidx" --vocab_size 65536 --model_type $model_type \
 --ctx_len $ctx_len --train_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $layer --n_embd $emb --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1
