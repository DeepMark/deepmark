local function vgg(batchSize)
   local modelType = 'D'

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            features:add(nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1))
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   features:get(1).gradInput = nil

   local classifier = nn.Sequential()
   classifier:add(nn.View(512*7*7))
   classifier:add(nn.Linear(512*7*7, 4096))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 1000))

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model, {batchSize,3,224,224}
end

return vgg
