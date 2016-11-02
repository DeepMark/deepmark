require 'optim'
require 'cunn'

local unpack = unpack or table.unpack
local pl = require('pl.import_into')()
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = pl.lapp[[
   --network (default "") network architecture: alexnet | inceptionv3 | resnet | vgg_d
   --backend (default "cudnn") backend type: cunn | cudnn
   --batchSize (default 128) batch size
   --dryrun  (default 10) number of iterations of a dry run not counted towards final timing
   --iterations (default 10) number of iterations to time and average over
   --nGPU (default 1) number of GPUs
]]

local net, isize = require(opt.network)(opt.batchSize)

-- randomly initialized input
local sizeTensor = torch.LongTensor(isize)
local optsizeTensor = sizeTensor:clone()
optsizeTensor[1] = 4
local input = torch.randn(sizeTensor:storage())
local dummyinput = torch.randn(optsizeTensor:storage())

-- random target classes from  1 to 100
local target = torch.Tensor(input:size(1)):random(1, 100)

-- cast network
net:cuda()
if opt.nGPU == 1 then
  input = input:cuda()
else
  local tmp = cutorch.createCudaHostTensor()
  tmp:resize(input:size()):copy(input)
  input = tmp
end 
if target.cudaLong then target = target:cudaLong() else target = target:cuda() end
dummyinput = dummyinput:cuda()

   

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
   cudnn.verbose = false
   net = cudnn.convert(net, cudnn)
end

-- optimize memory
optnet.optimizeMemory(net, dummyinput, {inplace=true, mode='training', reuseBuffers=true})

if opt.nGPU > 1 then
   assert(opt.nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
   if opt.network == 'alexnet' or opt.network == 'vgg_d' or opt.network == 'c3d' then
      local features_single = net:get(1)
      local features = nn.DataParallelTable(1, true, true)
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark
      features:threads(function()
   	    require 'cunn'
   	    require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
      end)
      local gpuid = torch.range(1, opt.nGPU):totable()
      features:add(features_single, gpuid)
      features.gradInput = nil
      net.modules[1] = features
   else
      net_single = net
      net = nn.DataParallelTable(1, true, true)
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark
      net:threads(function()
   	    require 'cunn'
   	    require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
      end)
      local gpuid = torch.range(1, opt.nGPU):totable()
      net:add(net_single, gpuid)
      net.gradInput = nil
   end
end


local params, gradParams = net:getParameters()

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
   cutorch.synchronizeAll()
   collectgarbage()
end

tm:reset()
for t = 1, opt.iterations do
   optim.sgd(feval, params, optimState)
   cutorch.synchronizeAll()
end
local time_taken_per_iter = tm:time().real / opt.iterations
local examples_per_sec = 1 / time_taken_per_iter * opt.batchSize 

print(string.format("Device: %1d x %-15s    Network: %-15s      Backend: %-10s      " ..
                    "Batchsize/GPU: %-5d      Iter (ms): %-10.2f Examples/sec: %-5d",
                    opt.nGPU, cutorch.getDeviceProperties(cutorch.getDevice()).name,
                    opt.network,
                    opt.backend,
                    opt.batchSize,
                    time_taken_per_iter * 1000,
                    examples_per_sec))
