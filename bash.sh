# for cifar100 
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 8 \
    --network.checkpoint 'checkpoint/resnet18_cifar 100.ckpt' \
    --postprocessor.postprocessor_args.K1 10 \
    --postprocessor.postprocessor_args.K2 5 \
    --postprocessor.postprocessor_args.ALPHA 0.5 \
    --postprocessor.postprocessor_args.queue_size 512 \
    --merge_option merge


# for cifar10 
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 8 \
    --network.checkpoint 'checkpoint/resnet18_cifar10.ckpt' \
    --postprocessor.postprocessor_args.K1 5 \
    --postprocessor.postprocessor_args.K2 5 \
    --postprocessor.postprocessor_args.ALPHA 0.5 \
    --postprocessor.postprocessor_args.queue_size 128 \
    --merge_option merge

# for imagenet200 
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'checkpoint/resnet18_imagenet200.ckpt' \
    --postprocessor.postprocessor_args.K1 100 \
    --postprocessor.postprocessor_args.K2 5 \
    --postprocessor.postprocessor_args.ALPHA 0.5 \
    --postprocessor.postprocessor_args.queue_size 2048 \
    --merge_option merge



# for imagenet1k 
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_myood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'checkpoint/resnet50_imagenet1k.pth' \
    --postprocessor.postprocessor_args.K1 100 \
    --postprocessor.postprocessor_args.K2 5 \
    --postprocessor.postprocessor_args.ALPHA 0.5 \
    --postprocessor.postprocessor_args.queue_size 2048 \
    --merge_option merge



