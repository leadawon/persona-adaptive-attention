torchrun --nproc_per_node 2 test.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 2048 --max_batch_size 8