unpack = unpack or table.unpack

require 'nn'
-- require 'cunn'
require 'paths'
require 'torch'
-- require 'cutorch'
require 'image'
-- require 'stn'
-- require 'BilinearSamplerBHWD'
require 'optim'
require 'ConvLSTM'
-- require 'display_flow'

--torch.setdefaulttensortype('torch.FloatTensor')
-------- build model

require 'nn'
require 'rnn'
require 'ConvLSTM'
require 'ConvLSTM_NoInput'
require 'SpatialConvolutionNoBias'


  dofile('./hko/opts-hko.lua')    
--  dofile('./hko/data-hko.lua')

-- sample = datasetSeq[3] --
data = torch.randn(7, 10, 4, 5, 5):cuda() -- :fill(torch.randn(0,1))
inputTable = {}
opt.nFiltersMemory[1] = 4
opt.nFiltersMemory[2] = 6
opt.input_nSeq = 3
opt.output_nSeq = 5
opt.batchSize = 7
for i = 1, opt.input_nSeq do
  table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1))
end
print("input table",inputTable)
-- reshape = nn.Sequential():add(nn.Reshape(4, 50, 50))
s = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
                  opt.input_nSeq, opt.kernelSize,
                  opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                  true, 3))
encoder_0 = s:cuda()
-- print(encoder_0)
output0 = s:updateOutput(inputTable)
print(encoder_0.outputs)

encoder_1 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
                  opt.input_nSeq, opt.kernelSize,
                  opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                  true, 3, true )):cuda() -- with cell 2 gate, kernel size 3

encoder_1:remember('both')
encoder_1:training()
output1 = encoder_1:updateOutput(output0)

------------ middle3, input size 64 to 64 ------------
------------ input is the output of decoder_2, copy cell&hid from encoder_1
decoder_3 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 5, 15?
                              opt.output_nSeq, opt.kernelSize,
                              opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                              true, 3, true )) :cuda()-- with cell 2 gate, kernel size 3
decoder_3:remember('both')
decoder_3:training()





-- print(output1)
-- print('###########################', encoder_1.module.outputs)
decoder_2 = nn.Sequencer(nn.ConvLSTM_NoInput(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 64, 64 inputSize and outputSize
                                      opt.output_nSeq,             -- length of seq
                                      0, opt.kernelSizeMemory,     -- size of kernel for intput2gate and memory2gate
                                      opt.stride, opt.batchSize,   -- stride and batchsize
                                      true, 3, -- with previous cell to gate, kernel size 3
                                      false)):cuda()  -- no input for LSTM

---- TODO: SET THE input of decoder 2 to be zeroTendor
---- TODO: set the kernel size of the decoder LSEM kernel size which multiple input to be 0?
decoder_2:remember('both')
decoder_2:training()

   decoder_2.module.userPrevOutput = nn.rnn.recursiveCopy(decoder_2.module.userPrevOutput, encoder_0.module.outputs[opt.input_nSeq])
   decoder_2.module.userPrevCell = nn.rnn.recursiveCopy(decoder_2.module.userPrevCell, encoder_0.module.cells[opt.input_nSeq])

   decoder_3.module.userPrevOutput = nn.rnn.recursiveCopy(decoder_3.module.userPrevOutput, encoder_1.module.outputs[opt.input_nSeq])
   decoder_3.module.userPrevCell = nn.rnn.recursiveCopy(decoder_3.module.userPrevCell, encoder_1.module.cells[opt.input_nSeq])

--      forwardConnect(encoder_1, decoder_3)
-- print(inputTable2)
-- print((inputTable2[2]))
inputTable2 = {decoder_2.module.userPrevOutput, decoder_2.module.userPrevCell}
-- print(decoder_2.module)
myzero = torch.Tensor(opt.batchSize, opt.nFiltersMemory[2], 5, 5):fill(0):cuda()
ini = {}
for i = 1, opt.output_nSeq do
    table.insert(ini, myzero)
end
output2 = decoder_2:updateOutput(ini)
print("haha",decoder_2.module.outputs)

output3 = decoder_3:updateOutput(output2)
print(output3)

concat = nn.ConcatTable()
for i = 1, 6 do
    s1 = nn.SelectTable(i)
    s2 = nn.SelectTable(i + 5)
    con = nn.ConcatTable():add(s1):add(s2)
    f = nn.Sequential():add(con):add(nn.JoinTable(2,4))
    concat:add(f)
end

flat1 = nn.ParallelTable():add(nn.FlattenTable()):add(nn.FlattenTable())

interface = nn.Sequential():add(flat1):add(nn.FlattenTable()):add(concat)

convForward_4 = interface:add(nn.Sequencer(nn.SpatialConvolution(2*opt.nFiltersMemory[2], opt.nFiltersMemory[1],
                                            3, 3,
                                            1,1,
                                            1, 1))):cuda()
inputTable4 = {{{output0[opt.input_nSeq]}, output2},{{output1[opt.input_nSeq]}, output3}}
pp = convForward_4:forward(inputTable4)
print(pp)

-- encoder_0.module.outputs[opt.input_nSeq]:size()
-- decoder_2.module.userPrevOutput:size()


-- print(convForward_4)
-- output = convForward_4:updateOutput(inputTable4)
-- print(inputTable4)
-- output = convForward_4:updateOutput(inputTable4)
-- print(output[1]:size())

-- criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())