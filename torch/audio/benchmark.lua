require 'sys'
require 'cunn'
require 'cudnn'
require 'nnx' -- For the CTCCriterion
require 'BatchBRNNReLU'

local Dataset = require 'Dataset'

cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
cudnn.verbose = false

local nets = {}
nets[#nets+1] = require 'cudnn_deepspeech2'

local libs = {}
libs[#libs+1] = {cudnn.SpatialConvolution, cudnn.ClippedReLU, cudnn.SpatialBatchNormalization, 'BDHW', 'cudnn'}

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

steps = 10 -- nb of steps in loop to average perf
nDryRuns = 10
batchSize = 20
spectrogramSize = 161
criterion = nn.CTCCriterion():cuda()

for i=1,#nets do
    for j=1,#libs do
        collectgarbage()
        local model,model_name = nets[i](libs[j])
        model = model:cuda()
        local input, target, labelLengths = Dataset.createDataset(batchSize, spectrogramSize)
        input = input:cuda()
        local lib_name = libs[j][5]
        print('ModelType: ' .. model_name, 'Kernels: ' .. lib_name,
            'Input shape: ' .. input:size(1) .. 'x' .. input:size(2) ..
                    'x' .. input:size(3) .. 'x' .. input:size(4))

        -- dry-run
        for i=1,nDryRuns do
            model:zeroGradParameters()
            local output = model:updateOutput(input)
            local gradInput = model:updateGradInput(input, output)
            model:accGradParameters(input, output)
            cutorch.synchronize()
            collectgarbage()
        end

        local tmf, tmbi, tmbg = 0,0,0
        local ok = 1
        for t = 1,steps do
            sys.tic()
            -- Forward through model and then criterion.
            output = model:updateOutput(input)
            loss = criterion:updateOutput(output, target, labelLengths)
            tmf = tmf + sys.toc()
            cutorch.synchronize()

            -- Backwards (updateGradInput, accGradParameters) including the criterion.
            sys.tic()
            gradInput = model:updateGradInput(input, output)
            criterion:updateGradInput(output, target, labelLengths)
            tmbi = tmbi + sys.toc()
            cutorch.synchronize()

            collectgarbage()
            sys.tic()
            ok = pcall(function() model:accGradParameters(input, output) end)
            tmbg = tmbg + sys.toc()
            cutorch.synchronize()
        end
        tmf = tmf/steps
        tmbi = tmbi/steps
        tmbg = tmbi/steps
        print(string.format("%-30s %25s %10.2f", lib_name, ':updateOutput():', tmf*1000))
        print(string.format("%-30s %25s %10.2f", lib_name, ':updateGradInput():', tmbi*1000))

        if not ok then
            print(string.format("%-30s %25s %s", lib_name, ':accGradParameters():', 'FAILED!'))
        else
            print(string.format("%-30s %25s %10.2f", lib_name, ':accGradParameters():', tmbg*1000))
        end
        print(string.format("%-30s %25s %10.2f", lib_name, ':Forward:', (tmf)*1000))
        print(string.format("%-30s %25s %10.2f", lib_name, ':Backward:', (tmbi+tmbg)*1000))
        print(string.format("%-30s %25s %10.2f", lib_name, ':TOTAL:', (tmf+tmbi+tmbg)*1000))
        print()
    end
end

print('')