set -e

if ! luajit -loptnet -e ""
then
    luarocks install optnet
fi

export LD_LIBRARY_PATH=$(pwd)/nccl_nvidia/build/lib:$LD_LIBRARY_PATH

if ! luajit -lnccl -e ""
then
    echo "Could not find NVIDIA NCCL. Downloading and compiling it locally"
    rm -rf nccl_nvidia
    git clone https://github.com/NVIDIA/nccl nccl_nvidia
    pushd nccl_nvidia
    make CUDA_HOME=/usr/local/cuda lib
    popd
    luarocks install nccl
fi

DRY_RUN_COUNT=10
ITERATIONS_COUNT=10
DEFAULT_ARGS="--dryrun $DRY_RUN_COUNT --iterations $ITERATIONS_COUNT"

echo "1-GPU benchmarks"

export CUDA_VISIBLE_DEVICES=0
th benchmark.lua --network alexnet                     $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16  $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30  $DEFAULT_ARGS

echo "2-GPU benchmarks"

export CUDA_VISIBLE_DEVICES=0,1
th benchmark.lua --network alexnet                     $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16  $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30  $DEFAULT_ARGS

echo "4-GPU benchmarks"

export CUDA_VISIBLE_DEVICES=0,1,2,3
th benchmark.lua --network alexnet                     $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16  $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30  $DEFAULT_ARGS

echo "8-GPU benchmarks"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
th benchmark.lua --network alexnet                     $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16  $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32  $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30  $DEFAULT_ARGS
