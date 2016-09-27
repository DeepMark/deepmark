require 'sys'
require 'cunn'
require 'cudnn'
require 'nnx'
require 'BatchBRNNReLU'
require 'Dataset'
local pl = require('pl.import_into')()



cudnn.fastest = true
cudnn.benchmark = false -- set this false due to the varying input shape
cudnn.verbose = false

local function printMemory()
    local freeMemory, totalMemory = cutorch.getMemoryUsage()
    print("total Memory", totalMemory, "free Memory", freeMemory, "used", totalMemory-freeMemory)
end

local opt = pl.lapp[[
   --dryrun  (default 10) number of iterations of a dry run not counted towards final timing
   --nGPU (default 1) number of GPUs to run on
   --batchSize (default 64) batch size
   --steps (default 1) number of steps to average performance
   --useOptnet (default true) whether to use optnet package for memory optimization
]]

local nGPU = opt.nGPU

deepSpeech = require 'cudnn_deepspeech2'

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

local steps = opt.steps -- nb of steps in loop to average perf
local nDryRuns = opt.dryrun
local batchSize = opt.batchSize
spectrogramSize = 161
criterion = nn.CTCCriterion():cuda()
local dataset = nn.DeepSpeechDataset(batchSize)
collectgarbage()
local model, model_name, calculateInputSizes = deepSpeech(batchSize, dataset.freqBins, nGPU, opt.useOptnet)
  
local inputs = torch.CudaTensor() -- buffer for inputs
local sizes, input, targets = dataset:nextTorchSet()
input=input:view(opt.batchSize,1,spectrogramSize, -1)

model = model:cuda()
inputs:resize(input:size()):copy(input)

print('ModelType: ' .. model_name, 'Kernels: ' .. 'cuDNN')

for i = 1, nDryRuns do
    model:zeroGradParameters()
    local output = model:updateOutput(inputs)
    local gradInput = model:updateGradInput(inputs, output)
    model:accGradParameters(inputs, output)
    cutorch.synchronize()
    collectgarbage()
end

local tmfAvg, tmbiAvg, tRoundTripAvg = 0,0,0,0

local ok = 1
for t = 1, steps do
    local tmf, tmbi, tRoundTrip = 0, 0, 0, 0
    local roundTripTimer = torch.Timer()
    dataset = nn.DeepSpeechDataset(batchSize)
    local numberOfIterations = 0
    local sizes, input, targets = dataset:nextTorchSet()
    while (sizes ~= nil) do
        input=input:view(opt.batchSize,1,spectrogramSize, -1)
        inputs:resize(input:size()):copy(input)        
        sys.tic()
        -- Forward through model and then criterion.
        local output = model:updateOutput(inputs)
        loss = criterion:updateOutput(output, targets, calculateInputSizes(sizes))
        cutorch.synchronize()
        tmf = tmf + sys.toc()

        -- Backwards (updateGradInput, accGradParameters) including the criterion.
        sys.tic()
        grads = criterion:updateGradInput(output, targets)
        model:backward(inputs,output)
        cutorch.synchronize()
        tmbi = tmbi + sys.toc()
--        collectgarbage()
        sizes, input, targets = dataset:nextTorchSet()
        numberOfIterations = numberOfIterations + 1
        xlua.progress(numberOfIterations * batchSize, dataset.size)
    end
    -- Divide the times to work out average time for updateOutput/updateGrad/accGrad
    tmfAvg = tmfAvg + tmf / numberOfIterations
    tmbiAvg = tmbiAvg + tmbi / numberOfIterations
    -- Add time taken for round trip of 1 epoch
    tRoundTripAvg = tRoundTripAvg + roundTripTimer:time().real
end
local tmf = tmfAvg / steps
local tmbi = tmbiAvg / steps
local tRoundTrip = tRoundTripAvg / steps
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':forward (ms):', tmf * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':backward (ms):', tmbi * 1000))

print(string.format("%-30s %25s %10.2f", 'cuDNN', ':TOTAL (ms):', (tmf + tmbi) * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Samples processed:', dataset.size))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Samples per second:', dataset.size / tRoundTrip))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Seconds of audio processed per second:',  dataset.duration / tRoundTrip))

print(string.format("%-30s %25s %10.2f", 'cuDNN', ':EPOCH TIME (s):', tRoundTrip))
print()

print('')
