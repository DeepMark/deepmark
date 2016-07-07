------------------------------------------------------------------------
--[[ SequenceWise ]] --
-- Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
-- Allows handling of variable sequence lengths and minibatch sizes.
------------------------------------------------------------------------

local SequenceWise, parent = torch.class('nn.SequenceWise', 'nn.Container')

function SequenceWise:__init(module)
    parent.__init(self)

    self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
    self.view_out = nn.View(1, -1):setNumInputDims(2)

    local sequenceWise = nn.Sequential()

    sequenceWise:add(self.view_in)
    sequenceWise:add(module)
    sequenceWise:add(self.view_out)

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.module = sequenceWise
    -- so that it can be handled like a Container
    self.modules[1] = sequenceWise
end

function SequenceWise:updateOutput(input)
    local T, N = input:size(1), input:size(2)
    self.view_in:resetSize(T * N, -1)
    self.view_out:resetSize(T, N, -1)
    self.output = self.module:updateOutput(input)
    return self.output
end

function SequenceWise:updateGradInput(input, gradOutput)
    self.gradInput = self.module:updateGradInput(input, gradOutput)
    return self.gradInput
end

function SequenceWise:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(input, gradOutput, scale)
end

function SequenceWise:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function SequenceWise:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function SequenceWise:__tostring__()
    if self.module.__tostring__ then
        return torch.type(self) .. ' @ ' .. self.module:__tostring__()
    else
        return torch.type(self) .. ' @ ' .. torch.type(self.module)
    end
end