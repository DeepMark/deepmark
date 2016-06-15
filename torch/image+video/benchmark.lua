require 'optim'
require 'cunn'

local unpack = unpack or table.unpack
local pl = require('pl.import_into')()
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = pl.lapp[[
   --network (default "") network architecture: alexnet-owt | googlenetv3 | resnet50 | vgg-d
   --backend (default "cudnn") backend type: cunn | cudnn
   --dryrun  (default 10) number of iterations of a dry run not counted towards final timing
   --iterations (default 10) number of iterations to time and average over
]]

local net, isize = require(opt.network)()

-- randomly initialized input
local input  = torch.randn(unpack(isize))

-- random target classes from  1 to 100
local target = torch.Tensor(input:size(1)):random(1, 100)

-- optimize memory
-- optnet.optimizeMemory(net, input, {inplace=true, mode='training', reuseBuffers=true})

-- cast network
net:cuda()
input, target = input:cuda(), target:cuda()
if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
   cudnn.verbose = false
   net = cudnn.convert(net, cudnn)
end

local params, gradParams = net:getParameters()

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

local criterion = nn.CrossEntropyCriterion():cuda()

local function feval(params)
   gradParams:zero()
   local output = net:forward(input)
   local loss = criterion:forward(output, target)
   local dloss_doutput = criterion:backward(output, target)
   net:backward(input, dloss_doutput)
   return loss, gradParams
end

-- no momentum and other terms for SGD
local optimState = {
   learningRate = 0.0001
}

local tm = torch.Timer()

-- dry-run
for i = 1, opt.dryrun do
   optim.sgd(feval, params, optimState)
   cutorch.synchronize()
   collectgarbage()
end

tm:reset()
for t = 1, opt.iterations do
   optim.sgd(feval, params, optimState)
   cutorch.synchronize()
end
local time_taken = tm:time().real / opt.iterations

print(string.format("Arch: %15s      Backend: %10s %25s %10.2f",
                    opt.network, opt.backend, ':TOTAL (ms) :',
                    time_taken * 1000))
print('')
