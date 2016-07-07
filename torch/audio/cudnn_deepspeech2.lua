-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 20) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 10) / 2 + 1) -- conv2
    return sizes
end

local function cudnnDeepSpeech2(miniBatchSize, freqBins, nGPUs)

    local model = nn.Sequential()
    model:add(nn.View(miniBatchSize / nGPUs, 1, freqBins, -1))
    model:add(cudnn.SpatialConvolution(1, 32, 20, 5, 2, 2))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ClippedReLU(true, 20))
    model:add(cudnn.SpatialConvolution(32, 32, 10, 5, 2, 1))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ClippedReLU(true, 20))

    model:add(nn.View(32 * 75, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    model:add(cudnn.BatchBRNNReLU(32 * 75, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))

    model:add(nn.SequenceWise(nn.Linear(1760, 29))) -- keeps the output 3D for multi-GPU.
    model = makeModelParallel(model, nGPUs)
    return model, 'cudnnDeepSpeech2', calculateInputSizes
end

function makeDataParallel(model, nGPU)
    if nGPU >= 1 then
        if nGPU > 1 then
            gpus = torch.range(1, nGPU):totable()
            dpt = nn.DataParallelTable(1):add(model, gpus):threads(function()
                local cudnn = require 'cudnn'
                cudnn.fastest = true
                require 'BatchBRNNReLU'
                require 'SequenceWise'
                require 'rnn'
            end)
            dpt.gradInput = nil
            model = dpt
        end
        model:cuda()
    end
    return model
end

return cudnnDeepSpeech2