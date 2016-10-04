local nn = require 'nn'
require 'cunn'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(batchSize)
   local depth = 50
   local shortcutType = 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   -- Configurations for ResNet:
   --  num. residual blocks, num features, residual block function
   local cfg = {
      [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      [152] = {{3, 8, 36, 3}, 2048, bottleneck},
   }

   assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
   local def, nFeatures, block = table.unpack(cfg[depth])
   iChannels = 64

   -- The ResNet ImageNet model
   model:add(Convolution(3,64,7,7,2,2,3,3))
   model:add(SBatchNorm(64))
   model:add(ReLU(true))
   model:add(Max(3,3,2,2,1,1))
   model:add(layer(block, 64, def[1]))
   model:add(layer(block, 128, def[2], 2))
   model:add(layer(block, 256, def[3], 2))
   model:add(layer(block, 512, def[4], 2))
   model:add(Avg(7, 7, 1, 1))
   model:add(nn.View(nFeatures):setNumInputDims(3))
   model:add(nn.Linear(nFeatures, 1000))

   model:get(1).gradInput = nil

   return model, {batchSize, 3, 224, 224}
end

return createModel
