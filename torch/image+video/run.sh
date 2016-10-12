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

th benchmark.lua --network alexnet                    --nGPU 1 $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16 --nGPU 1 $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32 --nGPU 1 $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32 --nGPU 1 $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30 --nGPU 1 $DEFAULT_ARGS

echo "2-GPU benchmarks"

th benchmark.lua --network alexnet                    --nGPU 2 $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16 --nGPU 2 $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32 --nGPU 2 $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32 --nGPU 2 $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30 --nGPU 2 $DEFAULT_ARGS

echo "4-GPU benchmarks"

th benchmark.lua --network alexnet                    --nGPU 4 $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16 --nGPU 4 $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32 --nGPU 4 $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32 --nGPU 4 $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30 --nGPU 4 $DEFAULT_ARGS

echo "8-GPU benchmarks"

th benchmark.lua --network alexnet                    --nGPU 8 $DEFAULT_ARGS
th benchmark.lua --network resnet      --batchSize 16 --nGPU 8 $DEFAULT_ARGS
th benchmark.lua --network vgg_d       --batchSize 32 --nGPU 8 $DEFAULT_ARGS
th benchmark.lua --network inceptionv3 --batchSize 32 --nGPU 8 $DEFAULT_ARGS
th benchmark.lua --network c3d         --batchSize 30 --nGPU 8 $DEFAULT_ARGS
