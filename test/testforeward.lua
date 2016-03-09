-- testforeward.lua
unpack = unpack or table.unpack

require 'nn'
require 'cunn'
require 'paths'
require 'torch'
require 'cutorch'
require 'image'
-- require 'stn'
-- require 'BilinearSamplerBHWD'
require 'optim'
require 'ConvLSTM'
-- require 'display_flow'

torch.setdefaulttensortype('torch.FloatTensor')
-------- build model

require 'nn'
require 'rnn'
require 'ConvLSTM'
require 'SpatialConvolutionNoBias'




function read_model(model)
    parameters, grads = model:getParameters()
    print('Number of parameters ' .. parameters:nElement())
    print('Number of grads ' .. grads:nElement())
end


  dofile('./hko/opts-hko.lua')
  dofile('./hko/data-hko.lua')


-- sample = datasetSeq[3] -- 
data = torch.randn(8, 20, 4, 50, 50)-- :fill(torch.randn(0,1))
inputTable = {}
for i = 1, input_seqlen do 
  table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1))
end

-- reshape = nn.Sequential():add(nn.Reshape(4, 50, 50))
s = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2],  -- 5, 15?
                  opt.input_nSeq, opt.kernelSize, 
                  opt.kernelSizeMemory, opt.stride, 8, -- batchsize
                  true, 3 ))
encoder_0 = s
output0 = s:updateOutput(inputTable)
print(output0:size())
