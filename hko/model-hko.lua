require 'nn'
require 'rnn'
require 'ConvLSTM'
require 'DenseTransformer2D'
require 'SmoothHuberPenalty'
require 'encoder'
require 'decoder'
require 'flow'
require 'stn'

model = nn.Sequential()

-- add encoder
-- local seqe = nn.Sequencer(encoder)
-- seqe:remember('both')
-- seqe:training()
-- model:add(seqe)

-- memory branch
local memory_branch = nn.Sequential()

------------ middle0, input size 4 to 64 -----------
local encoder_0 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2],  -- 5, 15?
									opt.nSeq, opt.kernelSize, 
									opt.kernelSizeMemory, opt.stride, 8 -- batchsize
									true, 3 )) -- with cell 2 gate, kernel size 3



encoder_0:remember('both')
encoder_0:training()

------------ middle1, input size 64 to 64 -----------
local encoder_1 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 5, 15?
									opt.nSeq, opt.kernelSize, 
									opt.kernelSizeMemory, opt.stride, 8 -- batchsize
									true, 3 )) -- with cell 2 gate, kernel size 3



encoder_1:remember('both')
encoder_1:training()

------------ middle2, input size 0 to 64 ------------
------------ (no input, copy cell&hid from encoder_0) 
local decoder_2 = nn.Sequencer(nn.ConvLSTMNoInput(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 5, 15?
									opt.nSeq, 0, -- size of kernel for input = 0
									opt.kernelSizeMemory, opt.stride, 8 -- batchsize
									true, 3 )) -- with cell 2 gate, kernel size 3



decoder_2:remember('both')
decoder_2:training()

------------ middle3, input size 64 to 64 ------------
------------ input is the output of decoder_2, copy cell&hid from encoder_1 
local decoder_3 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 5, 15?
									opt.nSeq, opt.kernelSize, 
									opt.kernelSizeMemory, opt.stride, 8 -- batchsize
									true, 3 )) -- with cell 2 gate, kernel size 3



decoder_3:remember('both')
decoder_3:training()
-------------- middle4, input size 64 to 4 ------------

local convForward_4 = nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 
											self.kc, self.kc, 
											self.stride, self.stride, 
											self.padc, self.padc)

--------------------------------------------------------
-- add connection between encoder_0 and decoder_2, encoder_1 and decoder_3
-- refer to https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua

local function forwardConnect(encLSTM, decLSTM)
   decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.seqLen])
   decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.seqLen])
end

local function backwardConnect(encLSTM, decLSTM)
   encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
   encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end

-----------------------------------------------------------


memory_branch:add(seq)

memory_branch:add(nn.SelectTable(opt.nSeq))



--memory_branch:add(nn.SelectTable(opt.nSeq))
--memory_branch:add(nn.L1Penalty(opt.constrWeight[2]))
-- memory_branch:add(flow)

-- keep last frame to apply optical flow on
-- local branch_up = nn.Sequential()
-- branch_up:add(nn.SelectTable(opt.nSeq))

-- transpose feature map for the sampler 
-- branch_up:add(nn.Transpose({1,3},{1,2}))

-- local concat = nn.ConcatTable()
-- concat:add(branch_up):add(memory_branch)
-- model:add(concat)

-- add sampler
-- model:add(nn.BilinearSamplerBHWD())
-- model:add(nn.Transpose({1,3},{2,3})) -- untranspose the result!!

-- add spatial decoder
-- model:add(decoder)

-- loss module: penalise difference of gradients
local gx = torch.Tensor(3,3):zero()
gx[2][1] = -1
gx[2][2] =  0
gx[2][3] =  1
gx = gx:cuda()

local gradx = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
gradx.weight:copy(gx)
gradx.bias:fill(0)

local gy = torch.Tensor(3,3):zero()
gy[1][2] = -1
gy[2][2] =  0
gy[3][2] =  1
gy = gy:cuda()

local grady = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
grady.weight:copy(gy)
grady.bias:fill(0)

local gradconcat = nn.ConcatTable()
gradconcat:add(gradx):add(grady)

gradloss = nn.Sequential()
gradloss:add(gradconcat)
gradloss:add(nn.JoinTable(1))

criterion = nn.MSECriterion()
--criterion.sizeAverage = false

-- move everything to gpu
model:cuda()
gradloss:cuda()
criterion:cuda()
  
