------------------------------------------------------------------------
--[[ Dataset ]] --
-- Creates a dataset of random numbers based on the Baidu chosen
-- distribution of dataset for the benchmark.
------------------------------------------------------------------------
local Dataset = {}

-- Each represents Length(s),Frequency(percent),Label length
local distribution = {
    { 1, 3.0, 7 },
    { 2, 10.0, 17 },
    { 3, 11.0, 35 },
    { 4, 13.0, 48 },
    { 5, 14.0, 62 },
    { 6, 13.0, 78 },
    { 7, 9.0, 93 },
    { 8, 8.0, 107 },
    { 9, 5.0, 120 },
    { 10, 4.0, 134 },
    { 11, 3.0, 148 },
    { 12, 2.0, 163 },
    { 13, 2.0, 178 },
    { 14, 2.0, 193 },
    { 15, 1.0, 209 },
}

local nbOfTimeStepsPerSecond = 100

-- When using batch sizes not of 100s, we use the largest remainder method to ensure that the dataset size is filled.
local function numberOfEntries(batchSize)
    local entries = {}
    for index, entry in ipairs(distribution) do
        local distribution = entry[2]
        local numberOfEntries = (distribution / 100) * batchSize
        entries[index] = numberOfEntries
    end
    -- Round everything down
    local flooredEntries = {}
    for index, entry in ipairs(entries) do
        flooredEntries[index] = math.floor(entry)
    end
    -- Calculate the decimal differences
    local decimals = {}
    for index, entry in ipairs(entries) do
        decimals[index] = entry- flooredEntries[index]
    end
    -- Add the index of entry before sorting it based on decimals
    local sortedEntries = {}
    for index, entry in ipairs(decimals) do
        sortedEntries[index] = { entry, index }
    end

    local function compare(a, b)
        return a[1] > b[1]
    end

    table.sort(sortedEntries, compare)

    local sumTensor = torch.Tensor(#distribution)

    for index, value in ipairs(entries) do
        sumTensor[index] = flooredEntries[index]
    end

    local counter = 1
    while (torch.sum(sumTensor) ~= batchSize) do
        local index = sortedEntries[counter][2]
        sumTensor[index] = sumTensor[index] + 1
    end
    return sumTensor
end

function Dataset.createDataset(batchSize, spectrogramSize)
    local maxLength = distribution[#distribution][1] * nbOfTimeStepsPerSecond
    local inputs = torch.Tensor(batchSize, 1, spectrogramSize, maxLength) -- we assume everything is padded with 0s.
    local targets = {}
    local labelLengths = torch.Tensor(batchSize)
    local counter = 1

    local numberOfEntries = numberOfEntries(batchSize)

    for index, entry in ipairs(distribution) do
        local seconds = entry[1]
        local distribution = entry[2]
        local labelLength = entry[3]
        local sequenceLength = seconds * nbOfTimeStepsPerSecond
        for x = 1, numberOfEntries[index] do
            local tensor = torch.randn(spectrogramSize, sequenceLength)
            inputs[counter][1]:narrow(2, 1, tensor:size(2)):copy(tensor) -- copy the tensor into fixed size dataset
            labelLengths[counter] = labelLength
            local target = torch.randn(labelLength)
            table.insert(targets, torch.totable(target))
            counter = counter + 1
        end
    end
    return inputs, targets, labelLengths
end

return Dataset