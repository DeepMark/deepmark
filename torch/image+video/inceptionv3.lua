require 'nn'

local function construct(batchSize)
   local json = require 'cjson'
   local f = io.open('inceptionv3.json')
   local config_str = f:read("*all")
   f:close()
   local config = json.decode(config_str)

   local function construct(v)
      local o = {}
      for k,vv in pairs(v) do
         if tonumber(k) then
            o[tonumber(k)] = vv
         else
            o[k] = vv
         end
      end
      v = o
      if v.type == 'inception' then
         local concat = nn.Concat(2)
         for i, m in ipairs(v) do
            local mod = construct(m)
            for kk, vv in ipairs(mod) do
               concat:add(vv)
            end
         end
         return {concat}
      elseif v.type == 'tower' then
         local sequential = nn.Sequential()
         for i, m in ipairs(v) do
            local mod = construct(m)
            for kk, vv in ipairs(mod) do
               sequential:add(vv)
            end
         end
         return {sequential}
      elseif v.type == 'conv2d' then
         return {
            nn.SpatialConvolution(v.iC, v.oC, v.kW, v.kH,
                                  v.strideW, v.strideH,
                                  v.padW, v.padH):noBias(),
            nn.SpatialBatchNormalization(v.oC, 0.0010000000475, nil, true),
            nn.ReLU(true)
         }
      elseif v.type == 'maxpool2d' then
         return {nn.SpatialMaxPooling(v.kH, v.kW,
                                      v.strideH, v.strideW,
                                      v.padH, v.padW)
         }
      elseif v.type == 'avgpool2d' then
         return {nn.SpatialAveragePooling(v.kH, v.kW,
                                          v.strideH, v.strideW,
                                          v.padH, v.padW)
         }
      else
         error('Unhandled type: ' .. v.type)
      end
   end

   local net = nn.Sequential()
   for k, v in ipairs(config) do
      local mod = construct(v)
      for k,v in ipairs(mod) do
         net:add(v)
      end
   end

   net:add(nn.View(-1):setNumInputDims(3))
   net:add(nn.Linear(2048, 1008))

   net.gradInput = nil

   return net, {batchSize, 3, 299, 299}
end

return construct
