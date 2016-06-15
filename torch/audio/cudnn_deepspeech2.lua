local function cudnnDeepSpeech2(lib)
    local SpatialConvolution = lib[1]
    local ReLU = lib[2]
    local SpatialBatchNormalization = lib[3]

    local model = nn.Sequential()
    model:add(SpatialConvolution(1, 32, 20, 5, 2, 2))
    model:add(SpatialBatchNormalization(32))
    model:add(ReLU(true, 20))
    model:add(SpatialConvolution(32, 32, 10, 5, 2, 1))
    model:add(SpatialBatchNormalization(32))
    model:add(ReLU(true, 20))

    model:add(nn.View(32 * 75, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    model:add(cudnn.BatchBRNNReLU(32 * 75, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))
    model:add(cudnn.BatchBRNNReLU(1760, 1760))

    model:add(nn.View(-1, 1760)) -- seqLength*batch x features (collapses into CTC format).
    model:add(nn.Linear(1760, 29))
    return model, 'cudnnDeepSpeech2'
end

return cudnnDeepSpeech2