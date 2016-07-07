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

steps = 10 -- nb of steps in loop to average perf
nDryRuns = 10
batchSize = 32
spectrogramSize = 161
criterion = nn.CTCCriterion():cuda()
local dataset = nn.DeepSpeechDataset(batchSize)

collectgarbage()
local model, model_name, calculateInputSizes = deepSpeech(batchSize, dataset.freqBins, nGPU)
local sizes, input, targets = dataset:nextTorchSet()

model = model:cuda()
input = input:cuda()
print('ModelType: ' .. model_name, 'Kernels: ' .. 'cuDNN')

-- dry-run
for i = 1, nDryRuns do
    model:zeroGradParameters()
    local output = model:updateOutput(input)
    local gradInput = model:updateGradInput(input, output)
    model:accGradParameters(input, output)
    cutorch.synchronize()
    collectgarbage()
end

local tmf, tmbi, tmbg = 0, 0, 0
local ok = 1
for t = 1, steps do
    dataset = nn.DeepSpeechDataset(batchSize)
    local numberOfIterations = 0
    local sizes, input, targets = dataset:nextTorchSet()
    while (sizes ~= nil) do
        input = input:cuda()
        sys.tic()
        -- Forward through model and then criterion.
        local output = model:updateOutput(input)
        loss = criterion:updateOutput(output, targets, calculateInputSizes(sizes))

        tmf = tmf + sys.toc()
        cutorch.synchronize()

        -- Backwards (updateGradInput, accGradParameters) including the criterion.
        sys.tic()
        grads = criterion:updateGradInput(output, targets)
        model:updateGradInput(input, output)
        tmbi = tmbi + sys.toc()
        cutorch.synchronize()

        collectgarbage()
        sys.tic()
        ok = pcall(function() model:accGradParameters(input, output) end)
        tmbg = tmbg + sys.toc()
        cutorch.synchronize()
        sizes, input, targets = dataset:nextTorchSet()
        numberOfIterations = numberOfIterations + 1
    end
    -- Divide the times to work out average time for updateOutput/updateGrad/accGrad
    tmf = tmf / numberOfIterations
    tmbi = tmbi / numberOfIterations
    tmbg = tmbi / numberOfIterations
end
tmf = tmf / steps
tmbi = tmbi / steps
tmbg = tmbi / steps
print(string.format("%-30s %25s %10.2f", lib_name, ':updateOutput():', tmf * 1000))
print(string.format("%-30s %25s %10.2f", lib_name, ':updateGradInput():', tmbi * 1000))

if not ok then
    print(string.format("%-30s %25s %s", lib_name, ':accGradParameters():', 'FAILED!'))
else
    print(string.format("%-30s %25s %10.2f", lib_name, ':accGradParameters():', tmbg * 1000))
end
print(string.format("%-30s %25s %10.2f", lib_name, ':Forward:', (tmf) * 1000))
print(string.format("%-30s %25s %10.2f", lib_name, ':Backward:', (tmbi + tmbg) * 1000))
print(string.format("%-30s %25s %10.2f", lib_name, ':TOTAL:', (tmf + tmbi + tmbg) * 1000))
print()

print('')