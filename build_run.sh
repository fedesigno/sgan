docker rmi eidos-service.di.unito.it/signoretta/sgan:sgcn_eth
docker build -t eidos-service.di.unito.it/signoretta/sgan:sgcn_eth . -f Dockerfile
docker push eidos-service.di.unito.it/signoretta/sgan:sgcn_eth

docker service rm signoretta-sgan-sgcn_eth
submit eidos-service.di.unito.it/signoretta/sgan:sgcn_eth train_SGCN_GEN.py \
  --dataset_name 'eth' \
  --delim tab \
  --d_type 'local' \
  --pred_len 12 \
  --encoder_h_dim_g 32 \
  --encoder_h_dim_d 64\
  --decoder_h_dim 32 \
  --embedding_dim 16 \
  --bottleneck_dim 32 \
  --mlp_dim 64 \
  --num_layers 1 \
  --noise_dim 8 \
  --noise_type gaussian \
  --noise_mix_type global \
  --pool_every_timestep 0 \
  --l2_loss_weight 1 \
  --batch_norm 0 \
  --dropout 0 \
  --batch_size 128 \
  --g_learning_rate 1e-3 \
  --g_steps 1 \
  --d_learning_rate 1e-3 \
  --d_steps 2 \
  --checkpoint_every 10 \
  --print_every 50 \
  --num_iterations 20000 \
  --num_epochs 500 \
  --pooling_type 'pool_net' \
  --clipping_threshold_g 1.5 \
  --best_k 10 \
  --gpu_num 1 \
  --checkpoint_name gan_test \
  --restore_from_checkpoint 0 \
  --dataset_dir '/data/trajgan/datasets/datasets_real' \
  --tag 'with_SGCN' \
  --dataset_dir_synth '/data/trajgan/datasets/datasets_synthetic' 

docker service logs -f signoretta-sgan-sgcn_eth

#  --dataset_dir_synth '/data/trajgan/datasets/datasets_sgcn_eth' \
