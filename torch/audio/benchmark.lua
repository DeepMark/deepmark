require 'sys'
require 'cunn'
require 'cudnn'
require 'nnx'
require 'BatchBRNNReLU'
require 'SequenceWise'
require 'Dataset'

cudnn.fastest = true
cudnn.benchmark = false -- set this false due to the varying input shape
cudnn.verbose = false
nGPU = 4

deepSpeech = require 'cudnn_deepspeech2'

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

steps = 1 -- nb of steps in loop to average perf
nDryRuns = 10
batchSize = 32
spectrogramSize = 161
criterion = nn.CTCCriterion():cuda()
local dataset = nn.DeepSpeechDataset(batchSize)

collectgarbage()
local model, model_name, calculateInputSizes = deepSpeech(batchSize, dataset.freqBins, nGPU)
local inputs = torch.CudaTensor() -- buffer for inputs
local sizes, input, targets = dataset:nextTorchSet()

model = model:cuda()
inputs:resize(input:size()):copy(input)

print('ModelType: ' .. model_name, 'Kernels: ' .. 'cuDNN')

-- dry-run
for i = 1, nDryRuns do
    model:zeroGradParameters()
    local output = model:updateOutput(inputs)
    local gradInput = model:updateGradInput(inputs, output)
    model:accGradParameters(inputs, output)
    cutorch.synchronize()
    collectgarbage()
end

local tmfAvg, tmbiAvg, tmbgAvg, tRoundTripAvg = 0,0,0,0

local ok = 1
for t = 1, steps do
    local tmf, tmbi, tmbg, tRoundTrip = 0, 0, 0, 0
    local roundTripTimer = torch.Timer()
    dataset = nn.DeepSpeechDataset(batchSize)
    local numberOfIterations = 0
    local sizes, input, targets = dataset:nextTorchSet()
    while (sizes ~= nil) do
        inputs:resize(input:size()):copy(input)
        sys.tic()
        -- Forward through model and then criterion.
        local output = model:updateOutput(inputs)
        loss = criterion:updateOutput(output, targets, calculateInputSizes(sizes))

        tmf = tmf + sys.toc()
        cutorch.synchronize()

        -- Backwards (updateGradInput, accGradParameters) including the criterion.
        sys.tic()
        grads = criterion:updateGradInput(output, targets)
        model:updateGradInput(inputs, output)
        tmbi = tmbi + sys.toc()
        cutorch.synchronize()

        collectgarbage()
        sys.tic()
        ok = pcall(function() model:accGradParameters(inputs, output) end)
        tmbg = tmbg + sys.toc()
        cutorch.synchronize()
        sizes, input, targets = dataset:nextTorchSet()
        numberOfIterations = numberOfIterations + 1
        xlua.progress(numberOfIterations * batchSize, dataset.size)
    end
    -- Divide the times to work out average time for updateOutput/updateGrad/accGrad
    tmfAvg = tmfAvg + tmf / numberOfIterations
    tmbiAvg = tmbiAvg + tmbi / numberOfIterations
    tmbgAvg = tmbgAvg + tmbg / numberOfIterations
    -- Add time taken for round trip of 1 epoch
    tRoundTripAvg = tRoundTripAvg + roundTripTimer:time().real
end
local tmf = tmfAvg / steps
local tmbi = tmbiAvg / steps
local tmbg = tmbgAvg / steps
local tRoundTrip = tRoundTripAvg / steps
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':updateOutput() (ms):', tmf * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':updateGradInput() (ms):', tmbi * 1000))

if not ok then
    print(string.format("%-30s %25s %s", 'cuDNN', ':accGradParameters() (ms):', 'FAILED!'))
else
    print(string.format("%-30s %25s %10.2f", 'cuDNN', ':accGradParameters() (ms):', tmbg * 1000))
end
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Forward (ms):', (tmf) * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Backward (ms):', (tmbi + tmbg) * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':TOTAL (ms):', (tmf + tmbi + tmbg) * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Samples processed:', dataset.size))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Samples per second:', dataset.size / tRoundTrip))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Seconds of audio processed per second:',  dataset.duration / tRoundTrip))

print(string.format("%-30s %25s %10.2f", 'cuDNN', ':EPOCH TIME (s):', tRoundTrip))
print()

print('')
