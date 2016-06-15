require 'cudnn'
require 'cunn'

local unpack = unpack or table.unpack
local pl = require('pl.import_into')()

torch.setdefaulttensortype('torch.FloatTensor')

local opt = pl.lapp[[
   --hidden  (default "{512, 1024}") number of hidden layers and each of their sizes
   --vocabsize (default 793471) number of words in the vocabulary
   --embedsize (default 512) size of each word embedding
   --batchsize (default 128) size of mini-batch
   --bptt      (default 20) number of timesteps to unroll for
   --backend (default "cudnn") backend type: cunn | cudnn
   --dryrun  (default 10) number of iterations of a dry run not counted towards final timing
   --iterations (default 10) number of iterations to time and average over
]]

local m = nn.Sequential()

m:add(nn.LookupTable(opt.vocabsize, opt.embedsize))
m:add(cudnn.LSTM(opt.embedsize, 8192, 1))
m:add(cudnn.LSTM(8192, 1024, 1))

m:cuda()

-- 128 sentences x 20 words
input = torch.LongTensor(opt.batchsize, opt.batchsize):random(1, opt.vocabsize):cuda()

output = m:forward(input)

print(#output)
