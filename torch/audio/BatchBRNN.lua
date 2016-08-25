------------------------------------------------------------------------
--[[ BiBatchRNN ]] --
-- Adds sequence-wise batch normalization to cudnn RNN modules.
-- Also adds bi-directionality to the chosen RNN.
-- For a simple RNN: ht = ReLU(B(Wixt) + Riht-1 + bRi) where B
-- is the batch normalization.
-- Expects size seqLength x minibatch x inputDim.
-- Returns seqLength x minibatch x outputDim.
-- Can specify an rnnModule such as cudnn.LSTM (defaults to RNNReLU).
------------------------------------------------------------------------
local BatchBRNN, parent = torch.class('cudnn.BatchBRNN', 'nn.Sequential')

function BatchBRNN:__init(inputDim, outputDim)
    parent.__init(self)

    self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
    self.view_out = nn.View(1, -1):setNumInputDims(2)

    self.rnn = cudnn.RNN(outputDim, outputDim, 1)
    local rnn = self.rnn
    rnn.inputMode = 'CUDNN_SKIP_INPUT'
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
    rnn.numDirections = 2
    rnn:reset()

    self:add(self.view_in)
    self:add(nn.Linear(inputDim, outputDim, false))
    self:add(cudnn.BatchNormalization(outputDim))
    self:add(self.view_out)
    self:add(rnn)
    self:add(nn.View(-1, 2, outputDim):setNumInputDims(2))
    self:add(nn.Sum(3))
end

function BatchBRNN:updateOutput(input)
    local T, N = input:size(1), input:size(2)
    self.view_in:resetSize(T * N, -1)
    self.view_out:resetSize(T, N, -1)
    return parent.updateOutput(self, input)
end

function BatchBRNN:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = ' -> '
    local str = 'BatchRNN'
    str = str .. ' {' .. line .. tab .. '[input'
    for i=1,#self.modules do
        str = str .. next .. '(' .. i .. ')'
    end
    str = str .. next .. 'output]'
    for i=1,#self.modules do
        str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
    end
    str = str .. line .. '}'
    return str
end