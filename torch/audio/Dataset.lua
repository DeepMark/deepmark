------------------------------------------------------------------------
--[[ Dataset ]] --
-- Creates a dataset of random numbers based on the Baidu chosen
-- distribution of dataset for the benchmark.
------------------------------------------------------------------------
require 'nn'
local Dataset = torch.class('nn.DeepSpeechDataset')

Dataset.uttLengths = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500 }
Dataset.counts = { 3, 10, 11, 13, 14, 13, 9, 8, 5, 4, 3, 2, 2, 2, 1 }
Dataset.label_lengths = { 7, 17, 35, 48, 62, 78, 93, 107, 120, 134, 148, 163, 178, 193, 209 }

Dataset.freqBins = 161

Dataset.scaleFactor = 10 * 128

Dataset.chars = 29

Dataset.extra = 1000

function Dataset:__init(minibatchSize)
    self.minibatchSize = minibatchSize
    self.current = 1
    self.uttCounts = {}
    for index, value in ipairs(Dataset.counts) do
        table.insert(self.uttCounts, value * Dataset.scaleFactor)
    end
    -- Swapped the freqBins around compared to ref for multiGPU support.
    self.randomness = torch.randn(minibatchSize * (Dataset.uttLengths[#Dataset.uttLengths] + self.extra), Dataset.freqBins)
    local size = 0
    local duration = 0
    for x =1, #Dataset.counts do
        size = size + Dataset.counts[x] * Dataset.scaleFactor -- get total count to track progress
        duration = duration + (Dataset.counts[x]  * Dataset.scaleFactor * Dataset.uttLengths[x]) / 100 -- get total duration of dataset (each second is 100 timesteps)
    end
    self.size = size
    self.duration = duration
end

function Dataset:next()
    if self.current > #self.uttCounts then
        return nil
    else
        local inc
        local miniBatchSize

        if (self.uttCounts[self.current] > self.minibatchSize) then
            miniBatchSize = self.minibatchSize
            self.uttCounts[self.current] = self.uttCounts[self.current] - self.minibatchSize
            inc = 0
        else
            miniBatchSize = self.uttCounts[self.current]
            self.uttCounts[self.current] = 0
            inc = 1
        end
        local uttLength = self.uttLengths[self.current]
        local labelLength = self.label_lengths[self.current]

        local startIndex = math.random(1, Dataset.extra + self.minibatchSize * (Dataset.uttLengths[#Dataset.uttLengths] - self.uttLengths[self.current]))

        local endIndex = startIndex + uttLength * miniBatchSize

        self.current = self.current + inc
        local label = torch.Tensor(labelLength)
        for x = 1, labelLength do
            label[x] = math.random(Dataset.chars)
        end
        local input = self.randomness[{{ startIndex, endIndex - 1 }, {} }]
        return uttLength, input, label
    end
end

-- Converts the iterator information into the format required for Torch benchmark
function Dataset:nextTorchSet()
    local uttLength, input, label = self:next()
    if (uttLength ~= nil) then
        -- Define length and label for each sample in minibatch.
        uttLength = torch.Tensor(self.minibatchSize):fill(uttLength)
        label = label:view(1, -1):expand(self.minibatchSize, label:size(1))
        return uttLength, input, torch.totable(label)
    else
        return nil
    end
end

return Dataset

