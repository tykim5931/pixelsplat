python -m src.main +experiment=dtu \
    wandb.mode=online \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=3 \
    dataset.view_selection_type=random \
    dataset.test_context_views=[23,24,22] \
    dataset.test_target_views=[34,14,32]


# 4 views gt pose
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=4 dataset.view_selection_type=random dataset.test_context_views=[23,25,33,15] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-4views-dtu-gtpose

# 3 views gt pose
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=3 dataset.view_selection_type=random dataset.test_context_views=[23,25,33] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-3views-dtu-gtpose

# 2 views gt pose
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=2 dataset.view_selection_type=random dataset.test_context_views=[23,25] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-2views-dtu-gtpose


# 4 views noisy pose (sigma 0.05)
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=4 dataset.view_selection_type=random dataset.test_context_views=[23,25,33,15] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-4views-dtu-noisy-005 test.noisy_pose=true test.noisy_level=0.05

# 3 views gt pose (sigma 0.03)
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=3 dataset.view_selection_type=random dataset.test_context_views=[23,25,33] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-3views-dtu-noisy-003 test.noisy_pose=true test.noisy_level=0.03

# 2 views gt pose (sigma 0.05)
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=2 dataset.view_selection_type=random dataset.test_context_views=[23,25] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-2views-dtu-noisy-005 test.noisy_pose=true test.noisy_level=0.05


# 4 views noisy pose (sigma 0.15)
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=4 dataset.view_selection_type=random dataset.test_context_views=[23,25,33,15] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-4views-dtu-noisy-015 test.noisy_pose=true test.noisy_level=0.15

# 3 views gt pose (sigma 0.15)
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=3 dataset.view_selection_type=random dataset.test_context_views=[23,24,33] dataset.test_target_views=[22,15,14] checkpointing.load=/mnt/nas4/youngju/ufosplat/pixelsplat/pixelsplat-dtu-3views.ckpt test.output_path=./test/dtu/005_depth test.noisy_pose=true test.noisy_level=0.05

# 2 views gt pose (sigma 0.15)
python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=2 dataset.view_selection_type=random dataset.test_context_views=[23,25] dataset.test_target_views=[34,14,32] checkpointing.load=/root/youngju/pixelsplat/checkpoints/pixelsplat-dtu-3views.ckpt test.output_path=./test/pixelsplat-2views-dtu-noisy-015 test.noisy_pose=true test.noisy_level=0.15



# re10k inference
python -m src.main +experiment=re10k wandb.mode=disabled mode=test data_loader.train.batch_size=1 checkpointing.load=checkpoints/pixelsplat_re10k.ckpt test.output_path=./test/re10k/depth_005 test.noisy_pose=true test.noisy_level=0.05 dataset/view_sampler=evaluation test.pred_pose_path=/mnt/nas4/youngju/ufosplat/ours/pred_mats_re10k_ours.pt


python -m src.main +experiment=dtu wandb.mode=disabled mode=test data_loader.train.batch_size=1 dataset.view_sampler.num_context_views=3 dataset.view_selection_type=random dataset.test_context_views=[34,14,32] dataset.test_target_views=[23,42,16] checkpointing.load=/mnt/nas4/youngju/ufosplat/pixelsplat/pixelsplat-dtu-3views.ckpt test.output_path=./test/dtu/005_large test.noisy_pose=true test.noisy_level=0.05