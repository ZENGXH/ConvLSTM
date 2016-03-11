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


SpatialConvolutionNoBias = SpatialConvolution
function nn.SpatialConvolutionNoBias(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        return nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
end



  dofile('./hko/opts-hko.lua')    
--  dofile('./hko/data-hko.lua')

-- sample = datasetSeq[3] --
data = torch.randn(7, 10, 4, 5, 5) -- :fill(torch.randn(0,1))
inputTable = {}
opt.nFiltersMemory[1] = 4
opt.nFiltersMemory[2] = 6
opt.input_nSeq = 3
opt.output_nSeq = 6
opt.batchSize = 7
opt.width = 5
    
for i = 1, opt.input_nSeq do
  table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1))
end


print("input table",inputTable)
-- reshape = nn.Sequential():add(nn.Reshape(4, 50, 50))
s = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
                  opt.input_nSeq, opt.kernelSize,
                  opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                  true, 3))
encoder_0 = s
-- print(encoder_0)
output0 = s:updateOutput(inputTable)
print(encoder_0.outputs)

encoder_1 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
                  opt.input_nSeq, opt.kernelSize,
                  opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                  true, 3, true )) -- with cell 2 gate, kernel size 3

encoder_1:remember('both')
encoder_1:training()
output1 = encoder_1:updateOutput(output0)

------------ middle3, input size 64 to 64 ------------
------------ input is the output of decoder_2, copy cell&hid from encoder_1
decoder_3 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 5, 15?
                              opt.output_nSeq, opt.kernelSize,
                              opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                              true, 3, true )) -- with cell 2 gate, kernel size 3
decoder_3:remember('both')
decoder_3:training()





-- print(output1)
-- print('###########################', encoder_1.module.outputs)
decoder_2 = nn.Sequencer(nn.ConvLSTM_NoInput(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 64, 64 inputSize and outputSize
                                      opt.output_nSeq,             -- length of seq
                                      0, opt.kernelSizeMemory,     -- size of kernel for intput2gate and memory2gate
                                      opt.stride, opt.batchSize,   -- stride and batchsize
                                      true, 3, -- with previous cell to gate, kernel size 3
                                      false))  -- no input for LSTM

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
myzero = torch.Tensor(opt.batchSize, opt.nFiltersMemory[2], 5, 5):fill(0)
ini = {}
for i = 1, opt.output_nSeq - 1 do
    table.insert(ini, myzero)
end

output2 = decoder_2:updateOutput(ini)
-- print("haha",decoder_2.module.outputs)

output3 = decoder_3:updateOutput(output2)
-- print(output3)

concat = nn.ConcatTable()
for i = 1, 6 do
    s1 = nn.SelectTable(i)
    s2 = nn.SelectTable(i + 5)
    con = nn.ConcatTable():add(s1):add(s2)
    f = nn.Sequential():add(con):add(nn.JoinTable(2,4))
    concat:add(f)
end

flat1 = nn.ParallelTable():add(nn.FlattenTable()):add(nn.FlattenTable())

inputTable4 = {{{output0[opt.input_nSeq]}, output2},{{output1[opt.input_nSeq]}, output3}}
interface = nn.Sequential():add(flat1):add(nn.FlattenTable()):add(concat)

in_out = interface:forward(inputTable4)

sequ = nn.Sequencer(nn.SpatialConvolution(2*opt.nFiltersMemory[2], opt.nFiltersMemory[1],
                                            3, 3,
                                            1,1,
                                            1, 1))
sequ:remember('both')
sequ:training()

convForward_4 = interface:add(sequ)
output = convForward_4:forward(inputTable4)

print("input to the sequcer\n", in_out, "\noutput ",pp) 
targetTable = {}
for i = 1, opt.output_nSeq do
  targetTable[i] = data[{{}, {opt.input_nSeq+i}, {}, {}, {}}]:select(2,1)
end
-- se:backward(in_out, {{torch.Tensor(7,4,5,5)},{torch.Tensor(7,4,5,5)},{torch.Tensor(7,4,5,5)},
-- {torch.Tensor(7,4,5,5)},{torch.Tensor(7,4,5,5)},{torch.Tensor(7,4,5,5)}})

--    targets = {torch.Tensor(7,4,5,5),torch.Tensor(7,4,5,5),
--        torch.Tensor(7,4,5,5),torch.Tensor(7,4,5,5),torch.Tensor(7,4,5,5),torch.Tensor(7,4,5,5)}
 -- gradO   criterion:backward(pp, targetTable)
criterion = nn.SequencerCriterion(nn.MSECriterion())
err = criterion:forward(output, torch.Tensor(6,7,4,5,5))

----- == -----
   gradOutput = criterion:backward(output, torch.Tensor(6,7,4,5,5))
   convForward_4:backward(inputTable4, gradOutput)
   
-- encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
-- encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
gardOutput_encoder_0 = convForward_4.gradInput[1][1]
gardOutput_encoder_1 = convForward_4.gradInput[2][1]
gardOutput_decoder_2 = convForward_4.gradInput[1][2]
gardOutput_decoder_3 = convForward_4.gradInput[2][2]
-- print("gradOutput copy done")
decoder_3:backward(output2, gardOutput_decoder_3)
-- print("backward decoder3, gardOutput\n",  gardOutput_decoder_2,"\ninput\n",output2,"\npass forward",decoder_3.gradInput)
assert(decoder_3.module.userGradPrevCell~=nil)

gardOutput_decoder_2 = {}
for i = 1, opt.output_nSeq - 1 do
    gardOutput_decoder_2[i] = decoder_3.gradInput[i] + convForward_4.gradInput[1][2][i]
    end
-- print("add grad output done",gardOutput_decoder_3)
decoder_2:backward(ini, gardOutput_decoder_2)

-------- backward connect:
encoder_0.module.userNextGradCell = nn.rnn.recursiveCopy(encoder_0.module.userNextGradCell, decoder_2.module.userGradPrevCell)
encoder_0.module.gradPrevOutput = nn.rnn.recursiveCopy(encoder_0.module.gradPrevOutput, decoder_2.module.userGradPrevOutput)

encoder_1.module.userNextGradCell = nn.rnn.recursiveCopy(encoder_1.module.userNextGradCell, decoder_3.module.userGradPrevCell)
encoder_1.module.gradPrevOutput = nn.rnn.recursiveCopy(encoder_1.module.gradPrevOutput, decoder_3.module.userGradPrevOutput)

assert(decoder_3.module.userPrevOutput~=nil)
assert(decoder_2.module.userGradPrevCell~=nil)
assert(decoder_3.module.gradPrevOutput~=nil)
assert(decoder_2.module.userGradPrevOutput~=nil)

-- print("connect backward done")

encoder_0:backward({inputTable[opt.input_nSeq]}, gardOutput_encoder_0)
encoder_1:backward({output0[opt.input_nSeq]}, gardOutput_encoder_1)




