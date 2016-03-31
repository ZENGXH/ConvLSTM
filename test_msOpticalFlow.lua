-- warp with optical flow
unpack = unpack or table.unpack
onMac = true
saveOutput = true
gpuflag = false
wrapOpticalFlow = false
underLowMemory = false

if wrapOpticalFlow then
	print('wrap optical Flow')
	imageDir = 'image/of/'
else
	imageDir = 'image/pure/'
end
print(imageDir)

local c = require 'trepl.colorize'

require 'modelSave'
require 'nn'
--require 'paths'
require 'torch'

if not onMac then
	require 'cunn'
	require 'cutorch'
	require 'stn'
	cutorch.setHeapTracking(true)
    cutorch.setDevice(1)
    local gpuid = 1
	local freeMemory, totalMemory = cutorch.getMemoryUsage(1)
	print('free',freeMemory/(1024^3), ' and total ', totalMemory/(1024^3))
	local freeMemory2, totalMemory = cutorch.getMemoryUsage(2)
	print('free2',freeMemory2/(1024^3), ' and total ', totalMemory/(1024^3))
	if(freeMemory2 > freeMemory) then 
		cutorch.setDevice(2)
		gpuid = 2
	end
end

require 'image'
-- require 'BilinearSamplerBHWD'
require 'optim'
require 'ConvLSTM'
require 'display_flow'
require 'DenseTransformer2D'
-- require 'flow'
-- torch.setdefaulttensortype('torch.FloatTensor')
-------- build model

require 'nn'
require 'rnn'
require 'ConvLSTM_NoInput'
local paraInit = 0.01
local std = 0.01

if onMac then
	print('on mac, SpatialConvolutionNoBias replace')
	function nn.SpatialConvolutionNoBias(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
	      return nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
	end
end


function checkMemory() 
  	if not onMac then
		local freeMemory, totalMemory = cutorch.getMemoryUsage(gpuid)
		print('free',freeMemory/(1024^3), ' and total ', totalMemory/(1024^3))
	end
end

dofile('./hko/opts-hko.lua')    
dofile('./hko/data-hko.lua')
dofile('./hko/data-valid.lua')
require 'flow'
print(opt)

if not onMac then
	data_path = '/csproject/dygroup2/xiaohui/ConvLSTM/helper/'
else
	data_path = '../ConvLSTM/helper/'	
end

datasetSeq= getdataSeq_mnist(data_path)
darasetSeq_valid = getdataSeq_valid(data_path)
print  ('Loaded ' .. datasetSeq:size() .. ' images')

print('==> training model')
checkMemory()
torch.manualSeed(opt.seed)  

------- ================
local eta0 = 1e-6
local eta = opt.eta
local errs= 0
local iter = 0
local epoch = 0
learningRate = 1e-5


if gpuflag then
	print('gpu flag on')
else
	print('gpu flag off')
end

encoder_0 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
                 		opt.input_nSeq, opt.kernelSize,
                        opt.kernelSizeMemory, opt.stride, opt.batchSize, true, 3)):float()-- batchsize


local lstm_params0, lstm_grads0 = encoder_0:getParameters()
lstm_params0:normal(paraInit, std)

----------------------------------

encoder_1 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
                  opt.input_nSeq, opt.kernelSize,
                  opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                  true, 3, true )):float() -- with cell 2 gate, kernel size 3

local lstm_params1, lstm_grads1 = encoder_1:getParameters()
lstm_params1:normal(paraInit, std)

print('encoder_1 build')
--checkMemory()
model = nn.Sequencer
-----------------------------------
--[[
decoder_2 = nn.Sequencer(nn.ConvLSTM_NoInput(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 64, 64 inputSize and outputSize
                                      opt.output_nSeq,             -- length of seq
                                      0, opt.kernelSizeMemory,     -- size of kernel for intput2gate and memory2gate
                                      opt.stride, opt.batchSize,   -- stride and batchsize
                                      true, 3, -- with previous cell to gate, kernel size 3
                                      false)):float()  -- no input for LST
]]
decoder_2 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 64, 64 inputSize and outputSize
                                      opt.output_nSeq,             -- length of seq
                                      opt.kernelSize,opt.kernelSizeMemory,     -- size of kernel for intput2gate and memory2gate
                                      opt.stride, opt.batchSize,   -- stride and batchsize
                                      true, 3, -- with previous cell to gate, kernel size 3
                                      true)):float()  -- no input for LST

local lstm_params2, lstm_grads2 = decoder_2:getParameters()
lstm_params2:normal(paraInit, std)


------------ middle3, input size 64 to 64 ------------
------------ input is the output of decoder_2, copy cell&hid from encoder_1
decoder_3 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 5, 15?
                              opt.output_nSeq, opt.kernelSize,
                              opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
                              true, 3, true )):float() -- with cell 2 gate, kernel size 3

local lstm_params3, lstm_grads3 = decoder_3:getParameters()
lstm_params3:normal(paraInit, std)

print('decoder_3 build')
--checkMemory()

-- store in CPU first
myzeroTensor = torch.Tensor(opt.batchSize, opt.nFiltersMemory[2], opt.width, opt.width):fill(0):float()

if gpuflag then
	myzeroTensor = myzeroTensor:cuda()
end
--assert(myzeroTensor:type() == "torch.cudaTensor")
--print('myzeroTensor type is ', myzeroTensor:type())


----------- build the block for convForward_4



flat1 = nn.ParallelTable():add(nn.FlattenTable()):add(nn.FlattenTable())
--  interface = nn.Sequential():add(flat1):add(nn.FlattenTable()):add(concat)


-- interface = nn.Sequential():add(nn.FlattenTable()):add(nn.JoinTable(1))
-- convForward_4 = interface:float()

reshape = nn.View(opt.nFiltersMemory[1], 100, 100) -- reshape to 2 * 100 * 100
-- one for inputs images , one for girds???

sequ = nn.Sequencer(nn.Sequential():add(nn.SpatialConvolution(opt.nFiltersMemory[2] * 2, 
										opt.nFiltersMemory[1], 
                                        3, 3, 1, 1, 1, 1))
                                    :add(reshape)
                                    :add(flow)):float()


--sequ:add(flow):add(reshape) -- depth = opt.nFiltersMemory, w,h = opt.memorySizeW = 50
--[[
memory_branch = nn.Sequential():float()
memory_branch:add(sequ)
-- originally: keep last frame to apply optical flow on

-- transpose feature map for the sampler 
local branch_up = nn.Sequential():add(nn.Sequencer(nn.Sequential()
                                                   :add(nn.View(opt.nFiltersMemory[2] * 2, opt.width, opt.width ))
                                                   :add(nn.SpatialConvolution(2 * opt.nFiltersMemory[2], opt.nFiltersMemory[1], 1, 1, 1, 1, 0, 0))
											       :add(nn.View(opt.imageDepth, opt.imageH, opt.imageW))
											       :add(nn.Transpose({1,3},{1,2})))):float()	  

-- originally: [depth, height, width]
-- after transpose [height, width, depth]

local memory_branch2 = sequ
local branch_up2 = nn.Sequencer(nn.Sequential()
                               :add(nn.View(opt.nFiltersMemory[2] * 2 , opt.width, opt.width ))
                               :add(nn.SpatialConvolution(opt.nFiltersMemory[2] * 2, opt.nFiltersMemory[1], 1, 1, 1, 1, 0, 0))
						       :add(nn.View(opt.imageDepth, opt.imageH, opt.imageW))
						       :add(nn.Transpose({1,3}, {1,2}))):float()
]]
local interface = nn.ConcatTable()

for i = 1, opt.output_nSeq do
    local s1 = nn.SelectTable(i)
    local s2 = nn.SelectTable(i + opt.output_nSeq)
    local con = nn.ConcatTable():add(s1):add(s2)
    local f = nn.Sequential():add(con):add(nn.JoinTable(2))
    interface:add(f)
end

local branch_up = nn.Sequential()
                               :add(nn.View(opt.nFiltersMemory[2] * 2 , opt.width, opt.width ))
                               :add(nn.SpatialConvolution(opt.nFiltersMemory[2] * 2, opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
						       :add(nn.View(opt.imageDepth, opt.imageH, opt.imageW))
						       :add(nn.Transpose({1,3}, {1,2})):float()
local memory_branch = nn.Sequential():add(nn.SpatialConvolution(opt.nFiltersMemory[2] * 2, 
										opt.nFiltersMemory[1], 
                                        3, 3, 1, 1, 1, 1))
                                    :add(reshape)
                                    :add(flow):float() 
--branch_up:add(nn.JoinTable(1)) -- along width direction
--memory_branch:add(nn.JoinTable(1)) -- along width direction
if not wrapOpticalFlow then
	print('not optical flow warpping')
	local concat = nn.ConcatTable()
	concat:add(branch_up):add(branch_up)
	wrapConcat = nn.Sequential():add(concat):add(nn.SelectTable(1)):add(nn.Transpose({1,3}, {2,3}))
	local sampler = nn.Sequencer(wrapConcat):float()
	convForward_4 = nn.Sequential():add(nn.FlattenTable()):add(interface):add(sampler):float()
--	convForward_4 = nn.Sequential():add(nn.FlattenTable()):add(sampler):float()

	convForward_4 = convForward_4:float()
	--	sampler = concat--:add(nn.CAddTable()):float()
--[[	sequencer_output2 = nn.Sequencer(nn.Sequential()
									:add(concat)
									:add(nn.CAddTable())
									:add(nn.Transpose({1,3}, {2,3}))
									)
--	sequencer_output3 = sequencer_output2:clone()
	
	convForward_4:add(nn.Sequencer(nn.Sequential()
									:add(concat)
									:add(nn.CAddTable())
									:add(nn.Transpose({1,3}, {2,3}))
									))]]

--	convForward_4 = nn.ParallelTable():add(sequencer_output2):add(sequencer_output3)

--	convForward_4:add(nn.FlattenTable()):add(nn.CAddTable())
				
	
--	sequencer_output2:add(nn.Sequential():add(nn.NarrowTable(4, 6)))
										--	:add(sequencer_output2))

--	convForward_4:add(nn.Sequencer(nn.Transpose({1,3}, {2,3}))):float()

--[[	convForward_4:add(nn.Transpose({1,3}, {2,3}))
           		 :add(nn.View(opt.imageDepth, opt.imageH, opt.imageW, opt.output_nSeq))
            	 :add(nn.SplitTable(4))]]
else
	local concat = nn.ConcatTable()
	concat:add(branch_up):add(memory_branch)
	wrapConcat = nn.Sequential():add(concat):add(nn.BilinearSamplerBHWD()):add(nn.Transpose({1,3}, {2,3}))
	local sampler = nn.Sequencer(wrapConcat):float()
    -- add sampler
    -- convForward_4:add(nn.BilinearSamplerBHWD())
--[[	convForward_4:add(nn.Sequencer(nn.Sequential()
									:add(nn.BilinearSamplerBHWD())
									:add(nn.Transpose({1,3}, {2,3}))))
]]
	convForward_4 = nn.Sequential():add(nn.FlattenTable()):add(interface):add(sampler):float()
	-- convForward_4 = convForward_4:float()

end
local lstm_params4, lstm_grads4 = convForward_4:getParameters()
lstm_params4:normal(paraInit, std)
--convForward_4:add(nn.Transpose({1,3}, {2,3}))
--             :add(nn.View(opt.imageDepth, opt.imageH, opt.imageW, opt.output_nSeq))

             
                  
----------------------------------------------------------------------------

print('convForward_4 build')
print(convForward_4)
--checkMemory()
local iter = 0

----- *********************************************************
----- **********************************************************
function train()
  epoch = epoch or 1  

  local epochSaveDir = imageDir..'train-epoch'..tostring(epoch)..'/'
  if not paths.dirp(epochSaveDir) then
   os.execute('mkdir -p ' .. epochSaveDir)
  end
  
  encoder_0:remember('both')
  encoder_0:training()
  encoder_1:remember('both')
  encoder_1:training()
  decoder_2:remember('both')
  decoder_2:training()
  decoder_3:remember('both')
  decoder_3:training()
  convForward_4:remember('both') 
  convForward_4:training()

  for t = 1, 2037 * 4 do

  	if testFlag and t == 2 then 
  		print('testing mode')
  		t = 2037 * 4 - 1
  		testFlag = false
  		print('exit testing mode')
  	end

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local tic = torch.tic()

    iter = t
--  local feval = function()
    local f = 0
    encoder_0:zeroGradParameters()
    encoder_0:forget()
    encoder_1:zeroGradParameters()
    encoder_1:forget()
    decoder_2:zeroGradParameters()
    decoder_2:forget()
    decoder_3:zeroGradParameters()
    decoder_3:forget()
    convForward_4:zeroGradParameters()
    convForward_4:forget()

    local sample = datasetSeq[t] 
    local data = sample[1]:float()

--   assert(data:size()[1] == 20 and data:nDimension() == 4, 'runnning optical_flow, ensure batchsize is 1')  
--	if gpuflag then data:cuda() end

	inputTable = {}

--	data:resize(opt.batchSize, opt.nSeq, opt.nFiltersMemory[1], opt.width, opt.width):float()
	data:resize(opt.batchSize, opt.nSeq, opt.imageDepth, opt.imageH, opt.imageW):float()
	
--	print(data:type())
--    print('input data size,', data:size())
--    print(data[{{}, {1}, {}, {}, {}}]:select(2,1))
    local  inputTable = {}

	for i = 1, opt.input_nSeq do
	    if opt.batchSize == 0 then
	    if gpuflag then
      	        table.insert(inputTable, data[{{i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
      	    else
      	    	table.insert(inputTable, data[{{i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
      	    end
      	else
      	    if gpuflag then
                table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
            else
            	table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
            end
        end
	end


	-- #todo: buffer parameters
	-- inputTable:cuda() 
	output0 = encoder_0:updateOutput(inputTable)
--	assert(output0[1] ~= nil)

	output1 = encoder_1:updateOutput(output0)
--	print('output1')
--[[	assert(output1[1] ~= nil)
	assert(encoder_0.module.outputs[opt.input_nSeq] ~= nil)
--	print(encoder_1.module.outputs)
	assert(encoder_1.module.outputs[opt.input_nSeq] ~= nil)
	assert(encoder_0.module.cells[opt.input_nSeq] ~= nil)
	assert(encoder_1.module.cells[opt.input_nSeq] ~= nil)

	decoder_2.module.userPrevOutput = nn.rnn.recursiveCopy(decoder_2.module.userPrevOutput, encoder_0.module.outputs[opt.input_nSeq])
	decoder_2.module.userPrevCell = nn.rnn.recursiveCopy(decoder_2.module.userPrevCell, encoder_0.module.cells[opt.input_nSeq])
	decoder_3.module.userPrevOutput = nn.rnn.recursiveCopy(decoder_3.module.userPrevOutput, encoder_1.module.outputs[opt.input_nSeq])
	decoder_3.module.userPrevCell = nn.rnn.recursiveCopy(decoder_3.module.userPrevCell, encoder_1.module.cells[opt.input_nSeq])
]]
	decoder_2.modules[1].outputs[1] = encoder_0.modules[1].outputs[opt.input_nSeq]:clone()
	
	decoder_2.modules[1].cells[1] = encoder_0.modules[1].cells[opt.input_nSeq]:clone()
	decoder_2.modules[1].cell = decoder_2.modules[1].cells[1]
	decoder_2.modules[1].output = decoder_2.modules[1].outputs[1]
	decoder_2.modules[1].step = 2

	decoder_3.modules[1].outputs[1] = encoder_1.modules[1].outputs[opt.input_nSeq]:clone()
	decoder_3.modules[1].cells[1] = encoder_1.modules[1].cells[opt.input_nSeq]:clone()
	decoder_3.modules[1].step = 2
	decoder_3.modules[1].cell = decoder_3.modules[1].cells[1]
	decoder_3.modules[1].output = decoder_3.modules[1].outputs[1]	
--[[
	assert(decoder_2.module.userPrevOutput ~= nil)
	assert(decoder_2.module.userPrevCell ~= nil)
	assert(decoder_3.module.userPrevOutput ~= nil)
	assert(decoder_3.module.userPrevCell ~= nil)
]]
--	print('decoder_2', decoder_2.module.userPrevOutput)
--	print('decoder_2', decoder_2.module.userPrevCell)

	-- inputTable2 = {decoder_2.module.userPrevOutput, decoder_2.module.userPrevCell}
	
	-- create fake input..
	ini = {}
	for i = 1, opt.output_nSeq - 1 do
		if gpuflag then
	        table.insert(ini, myzeroTensor:cuda())
	    else
	    	table.insert(ini, output0[opt.output_nSeq])
	    end
	end

--	print('\nforward to decoder_2====>')
--	checkMemory()
	
	decoder_2:updateOutput(ini)	
	output2 = decoder_2.modules[1].outputs

--  print(decoder_2.output)
--  print(decoder_2.modules[1].step)
--  assert(output2[1] ~= nil)
--	print('...')
--	checkMemory()
--	print('\nforward to decoder_3====>')
--	checkMemory()
	
	
	decoder_3:updateOutput(decoder_2.output)
	output3 = decoder_3.modules[1].outputs
--	assert(output3[1] ~= nil)
--	print('...')

	checkMemory()
--	print('\nforward to convForward_4====>')
	checkMemory()

--	inputTable4 = {{{output0[opt.input_nSeq]}, output2},{{output1[opt.input_nSeq]}, output3}}
	inputTable4 = {output2, output3}
if underLowMemory then
	print('underLowMemory, move all to float')
	checkMemory()
	encoder_0:float() 
	encoder_1:float() 
	decoder_2:float()
	decoder_3:float() 
	checkMemory()
end
--    print('input of convolution forward_4:')
--    print(inputTable4)

	output = convForward_4:forward(inputTable4)


--[[	
	print('output of CONVOLUTION forward, ')

if torch.isTensor(output) then 
		print(output:size())
	else
        print(output)
	end
]]
--        local opticalFlow = convForward_4.modules[4].output
--	print('4,2,1')
--	print('4,2,1,1,1,3,7')
--        print(convForward_4.modules[4].modules[2].modules[1].modules[1].modules[1].modules[3].modules[7].output)
    if saveOutput and math.fmod(t , 15) == 1 and t > 1 then
    	if wrapOpticalFlow then   
            local optical_flow = convForward_4.modules[4].modules[2].modules[1].modules[1].modules[1].modules[3].modules[7].output
    	    local imflow = flow2colour(optical_flow)
            image.save(epochSaveDir..'flow-iter'..tostring(iter)..'.png', imflow)
        end

        if torch.isTensor(output) then
            image.save(epochSaveDir..'output-iter'..tostring(iter)..'.png',  output)
        else
       	    for numsOfOut = 1, table.getn(output) do
       	    	local img = output[numsOfOut]:clone():float():resize(1,100,100)
	    	    image.save(epochSaveDir..'output-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end
	    	for numsOfOut = 1, table.getn(output2) do
       	    	local img = output2[numsOfOut]:clone():float():resize(1,100,100)
	    	    image.save(epochSaveDir..'output2-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end
	    	for numsOfOut = 1, table.getn(output3) do
       	    	local img = output3[numsOfOut]:clone():float():resize(1,100,100)
	    	    image.save(epochSaveDir..'output3-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end	    

	    	for numsOfOut = 1, table.getn(gradOutput) do
       	    	local img = gradOutput[numsOfOut]:clone():float():resize(1,1700,100)
       	    	img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..'gradOutput-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end
 
	    	for numsOfOut = 1, table.getn(convForward_4.gradInput[1]) do
       	    	local img = convForward_4.gradInput[1][numsOfOut]:clone():float():resize(1,1700,100)
       	    	img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..'convForward_4.gradInput[1]-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end	  

	    	for numsOfOut = 1, table.getn(convForward_4.gradInput[2]) do
       	    	local img = convForward_4.gradInput[2][numsOfOut]:clone():float():resize(1,1700,100)
       	    	img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..'convForward_4.gradInput[2]-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end  	    	

	    	for numsOfOut = 1, table.getn(output0) do
       	    	local img = output0[numsOfOut]:clone():float():resize(1,1700,100)
       	    	-- img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..'output0-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end 

	    	for numsOfOut = 1, table.getn(output1) do
       	    	local img = output1[numsOfOut]:clone():float():resize(1,1700,100)
       	    	-- img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..'output1-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end 
	    	
       	    	-- local img = output1[numsOfOut]:clone():float():resize(1,100,100)

	    	


        end
        print('image save')
    end

	criterion = nn.SequencerCriterion(nn.MSECriterion()):float()
	if gpuflag then criterion:cuda() end
     
   local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW):float()
   local target = {}
--	local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.nFiltersMemory[1], opt.width, opt.width):float()
	if opt.batchSize == 1 then
		for i = 1, opt.output_nSeq do
		  target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i])

		end
	else
		print('opt.batchSize > 1 !!')
		targetSeq[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:select(2,1):resizeAs(targetSeq[i])

	end
        if gpuflag then targetSeq = targetSeq:cuda() end

        if saveOutput and math.fmod(t , 15) == 1 and t > 1 then
	    	for numsOfOut = 1, table.getn(target) do
       	    	local img = target[numsOfOut]:clone():float():resize(1,100,100)
       	    	-- img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..'target-iter'..tostring(iter)..'n'..tostring(numsOfOut)..'.png',  img)
	    	end 
    	end

	err = criterion:forward(output, target)
	
	print("\titer",t, "err:", err)

	print('\ncriterion start bp <=======')  -- 1
	checkMemory()

	gradOutput = criterion:backward(output, targetSeq)

	print("...") -- 2
	checkMemory()
    targetSeq = nil
	convForward_4.outputs = {}

	--print(gradOutput)
	--checkMemory()

-------------------------------
	print('\nconv_4 bp <=======')  -- 1
	--checkMemory()

    convForward_4:backwardUpdate(inputTable4, gradOutput, learningRate)
    convForward_4:zeroGradParameters()

if underLowMemory then
	print('underLowMemory')
	checkMemory()
	encoder_0:float() 
	encoder_1:float() 
	decoder_2:float()
	decoder_3:float() 
	checkMemory()

end

	gardOutput_encoder_0 = {convForward_4.gradInput[1][1]}
	gardOutput_encoder_1 = {convForward_4.gradInput[2][1]}
	gardOutput_decoder_2 = nn.NarrowTable(2, opt.output_nSeq - 1):forward(convForward_4.gradInput[1])
	gardOutput_decoder_3 = nn.NarrowTable(2, opt.output_nSeq - 1):forward(convForward_4.gradInput[2])


    convForward_4:zeroGradParameters()
	convForward_4.output = {}


	--print("...forget")
	--checkMemory()
--------------------------------
	print('\ndecoder_3 bp <=======')  -- 1
	--checkMemory()

--	assert(gardOutput_decoder_3 ~= nil)
--	print(gardOutput_decoder_3)
--	print(decoder_2.output)
if underLowMemory then
	decoder_3:cuda()
	decoder_2:cuda()

end
	decoder_3:backward(decoder_2.output, gardOutput_decoder_3)
	decoder_3:updateParameters(learningRate)
 	print("...") -- 2
	--checkMemory()

--	assert(decoder_3.module.userGradPrevCell ~= nil)
--	assert(decoder_3.module.userGradPrevOutput ~= nil)
	-- copy for backward
	--gardOutput_decoder_2 = {}

	for i = 1, opt.output_nSeq - 1 do
--	    gardOutput_decoder_2[i] = gardOutput_decoder_2[i] + decoder_3.gradInput[i] 
	-- + convForward_4.gradInput[1][2][i] -- omit this one to save memory
	end
	
	-- print("add grad output done",gardOutput_decoder_3)
--	assert(decoder_3.module.userGradPrevCell ~= nil)
	-- before destory
if underLowMemory then
	encoder_1:cuda()
end
	-- decoder_3.userPrevCell = true

	encoder_1.module.gradCells[opt.input_nSeq] = nn.rnn.recursiveCopy(encoder_1.module.userGradPrevCell, 
														decoder_3.module.gradCells[1])
	encoder_1.module.gradPrevOutput = nn.rnn.recursiveCopy(encoder_1.module.gradPrevOutput, 
														decoder_3.module.gradPrevOutput[1])

--	decoder_3.module:clearAll_paraLeftOnly()
	decoder_3.module:forget()
	collectgarbage() -- 3
	print("...forget")
	--checkMemory()
--	print(gardOutput_encoder_1)
	encoder_1:backwardUpdate({output0[opt.input_nSeq]}, gardOutput_encoder_1, learningRate)
--	encoder_1.module:clearAll_paraLeftOnly()
	encoder_1.module:forget()
--	print('encoder_1 done bp')
----------------------------------

	-------- backward connect:
	--assert(decoder_2.module.userGradPrevCell~=nil)
	--assert(decoder_2.module.userGradPrevOutput~=nil)

	decoder_2:backwardUpdate(ini, gardOutput_decoder_2, learningRate)

	
--------------------------------
if underLowMemory then
	encoder_0:cuda()
end
	encoder_0.module.gradCells[opt.input_nSeq] = nn.rnn.recursiveCopy(encoder_0.module.userGradPrevCell, 
														decoder_2.module.gradCells[1])
	encoder_0.module.gradPrevOutput = nn.rnn.recursiveCopy(encoder_0.module.gradPrevOutput, 
														decoder_2.module.gradPrevOutput[1])

--	decoder_3.module:clearAll_paraLeftOnly()
	decoder_3.module:forget()
--	collectgarbage() -- 3
--	print("...forget")
	checkMemory()
	print(gardOutput_encoder_1)
	gardOutput_encoder_0 = {gardOutput_encoder_0[1] + encoder_1.module.gradInput}
	encoder_0:backwardUpdate({inputTable[opt.input_nSeq]}, gardOutput_encoder_0, learningRate)
--	encoder_1.module:clearAll_paraLeftOnly()
	encoder_0.module:forget()
--	print(encoder_1.modules[1].gradInput)
--	encoder_0:backwardUpdate(inputTable, encoder_1.module.gradInput,learningRate)
--	print('encoder_0 done bp')
	encoder_0.module:forget()

	ini = {}
	
--	print('decoder_2 bp done:')
	--checkMemory()
----------------------------------
--	print('encoder_1 bp done:')
	--checkMemory()

    
--	encoder_0:backwardUpdate({inputTable[opt.input_nSeq]}, gardOutput_encoder_0, learningRate)
--	encoder_0.module:clearAll_paraLeftOnly()
--	encoder_0.module:forget()
	convForward_4:forget()
--	print('encoder_0 bp done:')
	checkMemory()

	print("backward done")
	local toc = torch.toc(tic)
	print('time used: ',toc)
--    tic = torch.tic()
--    encoder_0:forget()
--    encoder_1:forget()
--    decoder_2:forget()
--    decoder_3:forget()
--    if math.fmod(t, 20 * opt.nSeq) == 1 then
 --   	local imflow = flow2colour(optical_flow)
        -- _im8_ = image.display{image=imflow,win=_im8_,legend='Flow'}
 --       image.save('image/flow-iter'..tostring(iter)..'.png', imflow)

        -- torch.save(opt.dir .. '/rmspropconf_' .. t .. '.bin', rmspropconf)
--   end

-- test
--[[   if opt.save and epoch == 0 and t == 3 then
   	   saveModel(encoder_0, 'encoder_0', t)
   	   print('0')
   	   checkMemory()
   	   saveModel(encoder_1, 'encoder_1', t)
   	   print('1')
   	   checkMemory()
   	   saveModel(decoder_2, 'decoder_2', t)
   	   print('2')
   	   checkMemory()
   	   saveModel(decoder_3, 'decoder_3', t)
   	   print('3')
   	   checkMemory()
   	   saveModel(convForward_4, 'convForward_4', t) 
   	   print('4')
   	   checkMemory()
   	   -- encoder_1, decoder_2, decoder_3, convForward_4, 
   	end

   -- if saveModel then
   if opt.save and math.fmod(t , 1000) == 1 and t > 1 then
   		print('model save')
--        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
       saveModel(encoder_0, 'encoder_0', t)
   	   saveModel(encoder_1, 'encoder_1', t)
   	   saveModel(decoder_2, 'decoder_2', t)
   	   saveModel(decoder_3, 'decoder_3', t)
   	   saveModel(convForward_4, 'convForward_4', t)
   end
]]
--  collectgarbage()
	checkMemory()
  end
epoch = epoch + 1
end


-- **************************************
function valid()
  local totalerror = 0
  local epochSaveDir = imageDir..'valid/'
  if not paths.dirp(epochSaveDir) then
   os.execute('mkdir -p ' .. epochSaveDir)
  end
  encoder_0:remember('eval')
  encoder_0:evaluate()
  encoder_1:remember('eval')
  encoder_1:evaluate()
  decoder_2:remember('eval')
  decoder_2:evaluate()
  decoder_3:remember('eval')
  decoder_3:evaluate()
  sequ:remember('eval') 
  sequ:evaluate()
  convForward_4:evaluate()

  for t = 1, 2037 do

  	if testFlag and t == 2 then 
  		print('testing mode')
  		t = 2037 - 1
  		testFlag = false
  		print('exit testing mode')
  	end

    print(c.blue '==>'..'testing')


    local iter = t
    local f = 0
    encoder_0:zeroGradParameters()
    encoder_0:forget()
    encoder_1:zeroGradParameters()
    encoder_1:forget()
    decoder_2:zeroGradParameters()
    decoder_2:forget()
    decoder_3:zeroGradParameters()
    decoder_3:forget()
    convForward_4:zeroGradParameters()
    convForward_4:forget()

    local sample = darasetSeq_valid[t] 
    local data = sample[1]:float()
--   assert(data:size()[1] == 20 and data:nDimension() == 4, 'runnning optical_flow, ensure batchsize is 1')  
--	if gpuflag then data:cuda() end
	inputTable = {}

--	data:resize(opt.batchSize, opt.nSeq, opt.nFiltersMemory[1], opt.width, opt.width):float()
	data:resize(opt.batchSize, opt.nSeq, opt.imageDepth, opt.imageH, opt.imageW):float()
	
--	print(data:type())
--    print('input data size,', data:size())
--    print(data[{{}, {1}, {}, {}, {}}]:select(2,1))
    local  inputTable = {}

	for i = 1, opt.input_nSeq do
	    if opt.batchSize == 0 then
	    if gpuflag then
      	        table.insert(inputTable, data[{{i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
      	    else
      	    	table.insert(inputTable, data[{{i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
      	    end
      	else
      	    if gpuflag then
                table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
            else
            	table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
            end
        end
	end


	-- #todo: buffer parameters
	-- inputTable:cuda() 
	output0 = encoder_0:updateOutput(inputTable)
--	assert(output0[1] ~= nil)

	output1 = encoder_1:updateOutput(output0)

	decoder_2.modules[1].outputs[1] = encoder_0.modules[1].outputs[opt.input_nSeq]:clone()
	
	decoder_2.modules[1].cells[1] = encoder_0.modules[1].cells[opt.input_nSeq]:clone()
	decoder_2.modules[1].cell = decoder_2.modules[1].cells[1]
	decoder_2.modules[1].output = decoder_2.modules[1].outputs[1]
	decoder_2.modules[1].step = 2

	decoder_3.modules[1].outputs[1] = encoder_1.modules[1].outputs[opt.input_nSeq]:clone()
	decoder_3.modules[1].cells[1] = encoder_1.modules[1].cells[opt.input_nSeq]:clone()
	decoder_3.modules[1].step = 2
	decoder_3.modules[1].cell = decoder_3.modules[1].cells[1]
	decoder_3.modules[1].output = decoder_3.modules[1].outputs[1]	
--[[
	assert(decoder_2.module.userPrevOutput ~= nil)
	assert(decoder_2.module.userPrevCell ~= nil)
	assert(decoder_3.module.userPrevOutput ~= nil)
	assert(decoder_3.module.userPrevCell ~= nil)
]]
--	print('decoder_2', decoder_2.module.userPrevOutput)
--	print('decoder_2', decoder_2.module.userPrevCell)

	-- inputTable2 = {decoder_2.module.userPrevOutput, decoder_2.module.userPrevCell}
	
	-- create fake input..
	ini = {}
	for i = 1, opt.output_nSeq - 1 do
		if gpuflag then
	        table.insert(ini, myzeroTensor:cuda())
	    else
	    	table.insert(ini, myzeroTensor)
	    end
	end

--	print('\nforward to decoder_2====>')
--	checkMemory()
	
	decoder_2:updateOutput(ini)	
	output2 = decoder_2.modules[1].outputs

--    print(decoder_2.output)
--    print(decoder_2.modules[1].step)
--    assert(output2[1] ~= nil)
--	print('...')
--	checkMemory()
--	print('\nforward to decoder_3====>')
--	checkMemory()
	
	
	decoder_3:updateOutput(decoder_2.output)
	output3 = decoder_3.modules[1].outputs
--	assert(output3[1] ~= nil)
--	print('...')

	checkMemory()
--	print('\nforward to convForward_4====>')
	checkMemory()

--	inputTable4 = {{{output0[opt.input_nSeq]}, output2},{{output1[opt.input_nSeq]}, output3}}
	inputTable4 = {output2, output3}
if underLowMemory then
	print('underLowMemory, move all to float')
	checkMemory()
	encoder_0:float() 
	encoder_1:float() 
	decoder_2:float()
	decoder_3:float() 
	checkMemory()
end
--    print('input of convolution forward_4:')
--    print(inputTable4)

	output = convForward_4:forward(inputTable4)

--   print(output)
--   print(targetSeq)
	local err = criterion:forward(output, targetSeq)
	totalerror = err + totalerror
	
end
    print("\tvalidation error", totalerror)
end

for k = 1, 200 do -- max epoch = 299

	if gpuflag then 
		checkMemory()
		print('move to GPU')
                encoder_0:cuda()
		encoder_1:cuda() 
		decoder_2:cuda()
		decoder_3:cuda() 
		convForward_4:cuda() 
		checkMemory()
	end

	if k == 1 then 
		print('test train ')
		checkMemory()
		testFlag = true
		train() 

		print('test valid')
		checkMemory()
		testFlag = true
	--	valid()

	   	print('test model save, move to CPU')
	   	checkMemory()
	--        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
                encoder_0:float()
		encoder_1:float() 
		decoder_2:float()
		decoder_3:float() 
		convForward_4:float() 
		checkMemory()
	   	saveModel(encoder_0, 'encoder_0', k)
	   	saveModel(encoder_1, 'encoder_1', k)
	   	saveModel(decoder_2, 'decoder_2', k)
	   	saveModel(decoder_3, 'decoder_3', k)
	   	saveModel(convForward_4, 'convForward_4', k)
	   	print('modelSave')
	   	checkMemory()
	end

		train()
		valid()

	if gpuflag then 
		encoder_0:float() 
		encoder_1:float() 
		decoder_2:float()
		decoder_3:float() 
		convForward_4:float() 
	    saveModel(encoder_0, 'encoder_0', k)
	   	saveModel(encoder_1, 'encoder_1', k)
	   	saveModel(decoder_2, 'decoder_2', k)
	   	saveModel(decoder_3, 'decoder_3', k)
	   	saveModel(convForward_4, 'convForward_4', k)
	   	print('modelSave')
	end

end
