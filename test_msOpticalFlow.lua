-- warp with optical flow
unpack = unpack or table.unpack
onMac = true
saveOutput = true
resumeFlag = false
resumeIter = 1

if onMac then
	gpuflag = false
	wrapOpticalFlow = false
else
	gpuflag = true
	wrapOpticalFlow = true
end


underLowMemory = false

if wrapOpticalFlow then
	print('wrap optical Flow')
	imageDir = 'image/of/'
else
	imageDir = 'image/pure/'
end
print(imageDir)

local c = require 'trepl.colorize'

require 'torch_extra/modelSave'

require 'nn'
require 'torch'


local saveInterval = 10

if not onMac then
	require 'cunn'
	require 'cutorch'
	require 'stn'
	cutorch.setHeapTracking(true)
    cutorch.setDevice(1)
    local gpuid = 1
	local freeMemory, totalMemory = cutorch.getMemoryUsage(1)
	print('free',freeMemory/(1024^3))
	local freeMemory2, totalMemory = cutorch.getMemoryUsage(2)
	print('free2',freeMemory2/(1024^3))
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
dofile('./hko/opts-hko.lua')    
dofile('./hko/data-hko.lua')
if not onMac then
	require 'flow'
end
print(opt)

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
		print('free',freeMemory/(1024^3))
	end
end


function saveImage(figure, name, iter, epochSaveDir, type)
	if torch.isTensor(figure) then
		local img = figure:clone():float():resize(opt.imageDepth, opt.imageH, opt.imageW)
	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-.png',  img/img:max())
	elseif type == 'output' then
    	for numsOfOut = 1, table.getn(figure) do
   	    	local img = figure[numsOfOut]:clone():float():resize(opt.imageDepth, opt.imageH, opt.imageW)
    	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
    	end 
	elseif type == 'gard' then
		for numsOfOut = 1, table.getn(figure) do
   	    	local img = figure[numsOfOut]:clone():float():resize(opt.imageDepth, opt.nFiltersMemory[2] * opt.imageH, opt.imageW)
   	    	img = img:mul(1/img:max())

    	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
    	end  
	end
end


if not onMac then
	data_path = '/csproject/dygroup2/xiaohui/ConvLSTM/helper/'
else
	data_path = '../ConvLSTM/helper/'	
end

datasetSeq= getdataSeq_hko('train', data_path)
darasetSeq_valid = getdataSeq_hko('valid', data_path)
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
local learningRate = 1e-5


if gpuflag then 
	print('gpu flag on') 
else 
	print('gpu flag off')
end

local iter = 0

if resumeFlag then
	local modelDir = 'image/'
	print('==> load model, resume traning')
  	encoder_0 = torch.load(modelDir..'test_encoder_0iter-'..resumeIter..'.bin')
  	encoder_1 = torch.load(modelDir..'test_encoder_1iter-'..resumeIter..'.bin')
  	decoder_2 = torch.load(modelDir..'test_decoder_2iter-'..resumeIter..'.bin')
  	decoder_3 = torch.load(modelDir..'test_decoder_3iter-'..resumeIter..'.bin')
  	convForward_4 = torch.load(modelDir..'convForward_4iter-'..resumeIter..'.bin')
  	print('load model done')
else
	print('==> build model for traning')

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
	-----------------------------------

	decoder_2 = nn.Sequencer(nn.ConvLSTM_NoInput(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 64, 64 inputSize and outputSize
	                                      opt.output_nSeq,             -- length of seq
	                                      0, opt.kernelSizeMemory,     -- size of kernel for intput2gate and memory2gate
	                                      opt.stride, opt.batchSize,   -- stride and batchsize
	                                      true, 3, -- with previous cell to gate, kernel size 3
	                                      false)):float()  -- no input for LST
	--[[
	decoder_2 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 64, 64 inputSize and outputSize
	                                      opt.output_nSeq,             -- length of seq
	                                      opt.kernelSize,opt.kernelSizeMemory,     -- size of kernel for intput2gate and memory2gate
	                                      opt.stride, opt.batchSize,   -- stride and batchsize
	                                      true, 3, -- with previous cell to gate, kernel size 3
	                                      true)):float()  -- no input for LST
	]]



	local lstm_params2, lstm_grads2 = decoder_2:getParameters()
	lstm_params2:normal(paraInit, std)
	-----------------------------------

	------------ middle3, input size 64 to 64 ------------
	------------ input is the output of decoder_2, copy cell&hid from encoder_1
	decoder_3 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[2],opt.nFiltersMemory[2],  -- 5, 15?
	                              opt.output_nSeq, opt.kernelSize,
	                              opt.kernelSizeMemory, opt.stride, opt.batchSize, -- batchsize
	                              true, 3, true )):float() -- with cell 2 gate, kernel size 3

	local lstm_params3, lstm_grads3 = decoder_3:getParameters()
	lstm_params3:normal(paraInit, std)

	--print('decoder_3 build')
	--checkMemory()
	-----------------------------------

	----------- build the block for convForward_4
	-- flat1 = nn.ParallelTable():add(nn.FlattenTable()):add(nn.FlattenTable())
	-- interface = nn.Sequential():add(flat1):add(nn.FlattenTable()):add(concat)
	-- interface = nn.Sequential():add(nn.FlattenTable()):add(nn.JoinTable(1))
	-- convForward_4 = interface:float()
	--[[
	-- originally: keep last frame to apply optical flow on

	-- transpose feature map for the sampler 

	-- originally: [depth, height, width]
	-- after transpose [height, width, depth]

	]]
	local interface = nn.ConcatTable()
	reshape = nn.View(opt.nFiltersMemory[1], opt.imageH, opt.imageW)

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
	if wrapOpticalFlow then						       
		local memory_branch = nn.Sequential():add(nn.SpatialConvolution(opt.nFiltersMemory[2] * 2, opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
	                                    :add(reshape)
	                                    :add(flow):float() 
    end
	--branch_up:add(nn.JoinTable(1)) -- along width direction
	--memory_branch:add(nn.JoinTable(1)) -- along width direction
	if not wrapOpticalFlow then
		print('not optical flow warpping')
		local concat = nn.ConcatTable()
		concat:add(branch_up):add(branch_up)
		local wrapConcat = nn.Sequential():add(concat):add(nn.SelectTable(1)):add(nn.Transpose({1,3}, {2,3}))
		local sampler = nn.Sequencer(wrapConcat):float()
		convForward_4 = nn.Sequential():add(nn.FlattenTable()):add(interface):add(sampler):float()
	--	convForward_4 = nn.Sequential():add(nn.FlattenTable()):add(sampler):float()
		convForward_4 = convForward_4:float()
	else
		local concat = nn.ConcatTable()
		concat:add(branch_up):add(memory_branch)
		local wrapConcat = nn.Sequential():add(concat):add(nn.BilinearSamplerBHWD()):add(nn.Transpose({1,3}, {2,3}))
		local sampler = nn.Sequencer(wrapConcat):float()
	    -- add sampler
		convForward_4 = nn.Sequential():add(nn.FlattenTable()):add(interface):add(sampler):float()
		-- convForward_4 = convForward_4:float()
	end
	local lstm_params4, lstm_grads4 = convForward_4:getParameters()
	lstm_params4:normal(paraInit, std)
	--convForward_4:add(nn.Transpose({1,3}, {2,3}))
	--             :add(nn.View(opt.imageDepth, opt.imageH, opt.imageW, opt.output_nSeq))
	----------------------------------------------------------------------------
end
--checkMemory()
-- store in CPU first
myzeroTensor = torch.Tensor(opt.batchSize, opt.nFiltersMemory[2], opt.width, opt.width):fill(0):float()
if gpuflag then myzeroTensor = myzeroTensor:cuda() end

print('encoder_0')
print(encoder_0)
print('encoder_1')
print(encoder_1)
print('decoder_2')
print(decoder_2)
print('decoder_3')
print(decoder_3)
print('convForward_4')
print(convForward_4)
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
  t = 0
  while t < 2037 * 4 do
  	t = t + 1
  	if testFlag and t == 2 then 
  		print('testing mode')
  		print('exit testing mode')
  		t = 2000 * 4 - 1
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

--	data:resize(opt.batchSize, opt.nSeq, opt.nFiltersMemory[1], opt.width, opt.width):float()
	data:resize(opt.batchSize, opt.nSeq, opt.imageDepth, opt.imageH, opt.imageW):float()
	
--	print(data:type())
--    print('input data size,', data:size())
--    print(data[{{}, {1}, {}, {}, {}}]:select(2,1))
    local  inputTable = {}

	for i = 1, opt.input_nSeq do
		if gpuflag then
	        table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
	    else
	    	table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
	    end
	end

	-- #todo: buffer parameters
	-- inputTable:cuda() 
	output0 = encoder_0:updateOutput(inputTable)

	output1 = encoder_1:updateOutput(output0)
--	assert(output0[1] ~= nil)
--	print('output1')

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

	--	print('decoder_2', decoder_2.module.userPrevOutput)
	--	print('decoder_2', decoder_2.module.userPrevCell)
	-- create fake input..
	local ini = {}
	for i = 1, opt.output_nSeq - 1 do
		if gpuflag then
--			print('zero input for decoder_2')
	        table.insert(ini, myzeroTensor:cuda())
	    else
--	    	print('output0 last frame input for decoder_2')
	    	table.insert(ini, output0[opt.output_nSeq])
	    end
	end

--	print('\nforward to decoder_2====>')
--	checkMemory()
	
	decoder_2:updateOutput(ini)	
	output2 = decoder_2.modules[1].outputs

	decoder_3:updateOutput(decoder_2.output)
	output3 = decoder_3.modules[1].outputs
	if t < 10 then checkMemory() end

--	inputTable4 = {{{output0[opt.input_nSeq]}, output2},{{output1[opt.input_nSeq]}, output3}}

	inputTable4 = {output2, output3}

if underLowMemory then
	print('underLowMemory, move all to float')
	if t < 10 then checkMemory() end
	encoder_0:float() 
	encoder_1:float() 
	decoder_2:float()
	decoder_3:float() 
	if t < 10 then checkMemory() end
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


	if testFlag and t == 2 then
    	if wrapOpticalFlow then   
            local optical_flow = convForward_4.modules[3].modules[1].modules[1].modules[1].modules[2].modules[3].modules[7].output
    	    local imflow = flow2colour(optical_flow)
            saveImage(imflow, 'flow', iter, epochSaveDir)
        end

        saveImage(output, 'output', iter, epochSaveDir, 'output')
        saveImage(output2, 'output2', iter, epochSaveDir, 'output')
        saveImage(output3, 'output3', iter, epochSaveDir, 'output')
        saveImage(output0, 'output0', iter, epochSaveDir, 'output')

        saveImage(convForward_4.gradInput[2], 'convForward_4.gradInput[2]', iter, epochSaveDir, 'grad')
        saveImage(convForward_4.gradInput[1], 'convForward_4.gradInput[1]', iter, epochSaveDir, 'grad')
        
        print('image save')
        testFlag = false
    end		
]]

    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	if wrapOpticalFlow then  
            local optical_flow = convForward_4.modules[3].modules[1].modules[1].modules[1].modules[2].modules[3].modules[7].output

    	    local imflow = flow2colour(optical_flow)
            saveImage(imflow, 'flow', iter, epochSaveDir)
        end

        saveImage(inputTable, 'inputTable', iter, epochSaveDir, 'output')

        saveImage(output, 'output', iter, epochSaveDir, 'output')
        saveImage(output2, 'output2', iter, epochSaveDir, 'output')
        saveImage(output3, 'output3', iter, epochSaveDir, 'output')
        saveImage(output0, 'output0', iter, epochSaveDir, 'output')

        saveImage(convForward_4.gradInput[2], 'convForward_4.gradInput[2]', iter, epochSaveDir, 'grad')
        saveImage(convForward_4.gradInput[1], 'convForward_4.gradInput[1]', iter, epochSaveDir, 'grad')
        
        print('image save')
    end
        --[[
        if torch.isTensor(output) then
            image.save(epochSaveDir..tostring(iter)..'-iter-output'..'.png',  output)
        else
       	    for numsOfOut = 1, table.getn(output) do
       	    	local img = output[numsOfOut]:clone():float():resize(1,100,100)
	    	    image.save(epochSaveDir..tostring(iter)..'output-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end

	    	for numsOfOut = 1, table.getn(output2) do
       	    	local img = output2[numsOfOut]:clone():float():resize(1,100,100)
	    	    image.save(epochSaveDir..tostring(iter)..'output2-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end
	    	for numsOfOut = 1, table.getn(output3) do
       	    	local img = output3[numsOfOut]:clone():float():resize(1,100,100)
	    	    image.save(epochSaveDir..tostring(iter)..'output3-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end	    

	    	for numsOfOut = 1, table.getn(gradOutput) do
       	    	local img = gradOutput[numsOfOut]:clone():float():resize(1,100,100)
       	    	img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..tostring(iter)..'gradOutput-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end
 
	    	for numsOfOut = 1, table.getn(convForward_4.gradInput[1]) do
       	    	local img = convForward_4.gradInput[1][numsOfOut]:clone():float():resize(1,1700,100)
       	    	img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..tostring(iter)..'convForward_4.gradInput[1]-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end	  

	    	for numsOfOut = 1, table.getn(convForward_4.gradInput[2]) do
       	    	local img = convForward_4.gradInput[2][numsOfOut]:clone():float():resize(1,1700,100)
       	    	img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..tostring(iter)..'convForward_4.gradInput[2]-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end  	    	

	    	for numsOfOut = 1, table.getn(output0) do
       	    	local img = output0[numsOfOut]:clone():float():resize(1,1700,100)
       	    	-- img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..tostring(iter)..'output0-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end 

	    	for numsOfOut = 1, table.getn(output1) do
       	    	local img = output1[numsOfOut]:clone():float():resize(1,1700,100)
       	    	-- img = img:mul(1/img:max())
	    	    image.save(epochSaveDir..tostring(iter)..'output1-iter'..'n'..tostring(numsOfOut)..'.png',  img)
	    	end 
	    ]]	
       	    	-- local img = output1[numsOfOut]:clone():float():resize(1,100,100)



	criterion = nn.SequencerCriterion(nn.MSECriterion()):float()
	if gpuflag then criterion:cuda() end
     
    local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW):float()
    local target = {}
--	local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.nFiltersMemory[1], opt.width, opt.width):float()
	if opt.batchSize == 1 then
		for i = 1, opt.output_nSeq do
            if gpuflag then
		        target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i]):cuda()
		    else
                target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i])
            end
		end

	else
		print('opt.batchSize > 1 !!')
--		targetSeq[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:select(2,1):resizeAs(targetSeq[i])
	end

--    if gpuflag then targetSeq = targetSeq:cuda() end
    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	saveImage(target, 'target', iter, epochSaveDir, 'output')
	end

	err = criterion:forward(output, target)
	
	print("\titer",t, "err:", err)
	if t < 10 then 
		checkMemory() 
		print('\ncriterion start bp <<<<')  -- 1
	end

	gradOutput = criterion:backward(output, target)

	print("...") -- 2
	if t < 10 then checkMemory() end
    target = {}
	convForward_4.outputs = {}

--	print(gradOutput)
	--checkMemory()

-------------------------------
	
	if t < 10 then 
		checkMemory() 
		print('\nconv_4 bp <<<<')  -- 1
	end

    convForward_4:backwardUpdate(inputTable4, gradOutput, learningRate)
    convForward_4:zeroGradParameters()
    convForward_4.output = {}

    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
        saveImage(gradOutput, 'gradOutput', iter, epochSaveDir, 'grad')
	end
	gradOutput = {}

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
--	print(convForward_4.modules)
--	convForward_4:forget()
--	print(convForward_4.modules)
	--print("...forget")
	--checkMemory()
--------------------------------
--	print('\ndecoder_3 bp <=======')  -- 1
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

-- 	print("...") -- 2
	--checkMemory()

--	assert(decoder_3.module.userGradPrevCell ~= nil)
--	assert(decoder_3.module.userGradPrevOutput ~= nil)
	-- copy for backward
	--gardOutput_decoder_2 = {}

	for i = 1, opt.output_nSeq - 1 do
	    gardOutput_decoder_2[i] = gardOutput_decoder_2[i] + decoder_3.gradInput[i] 
	-- + convForward_4.gradInput[1][2][i] -- omit this one to save memory
	end
	
	-- print("add grad output done",gardOutput_decoder_3)
--	assert(decoder_3.module.userGradPrevCell ~= nil)
	-- before destory
	if underLowMemory then encoder_1:cuda() end
	-- decoder_3.userPrevCell = true

	encoder_1.module.gradCells[opt.input_nSeq] = nn.rnn.recursiveCopy(encoder_1.module.userGradPrevCell, 
														decoder_3.module.gradCells[1])
	encoder_1.module.gradPrevOutput = nn.rnn.recursiveCopy(encoder_1.module.gradPrevOutput, 
														decoder_3.module.gradPrevOutput[1])
	
--	decoder_3.module:clearAll_paraLeftOnly()
	decoder_3:zeroGradParameters()
	decoder_3.module:forget()
	decoder_3.modules[1].cells = {}
	collectgarbage() -- 3
--	print("...forget")
	if t < 10 then checkMemory() end
--	print(gardOutput_encoder_1)
	encoder_1:backwardUpdate({output0[opt.input_nSeq]}, gardOutput_encoder_1, learningRate)
	if t < 10 then checkMemory() end
--	encoder_1.module:clearAll_paraLeftOnly()
	encoder_1.module:forget()
	encoder_1:zeroGradParameters()
	if t < 10 then checkMemory() end
	encoder_1.modules[1].cells = {}
--	print('encoder_1 done bp')
----------------------------------

	-------- backward connect:
	decoder_2:backwardUpdate(ini, gardOutput_decoder_2, learningRate)
	decoder_2.modules[1].cells = {}
--------------------------------
	if underLowMemory then encoder_0:cuda() end
	encoder_0.module.gradCells[opt.input_nSeq] = nn.rnn.recursiveCopy(encoder_0.module.userGradPrevCell, 
														decoder_2.module.gradCells[1])
	encoder_0.module.gradPrevOutput = nn.rnn.recursiveCopy(encoder_0.module.gradPrevOutput, 
														decoder_2.module.gradPrevOutput[1])
    decoder_2:forget()
    decoder_2:zeroGradParameters() 
    -- TODO: CHECK IF CLEAN GRAD OUTPUT AND CELLS

--	decoder_3.module:clearAll_paraLeftOnly()
--	decoder_3.module:forget()

	if t < 10 then checkMemory() end
--	print(gardOutput_encoder_1)
	gardOutput_encoder_0 = {gardOutput_encoder_0[1] + encoder_1.module.gradInput}
	encoder_0:backwardUpdate({inputTable[opt.input_nSeq]}, gardOutput_encoder_0, learningRate)
	encoder_0.module:forget()
	encoder_0:zeroGradParameters()

--	encoder_1.module:clearAll_paraLeftOnly()

--	print(encoder_1.modules[1].gradInput)
--	encoder_0:backwardUpdate(inputTable, encoder_1.module.gradInput,learningRate)
--	print('encoder_0 done bp')
	
	ini = {}
	inputTable =  {}
	
	convForward_4:forget()
--	print('encoder_0 bp done:')
	if t < 10 then checkMemory() end

	print("backward done")
	local toc = torch.toc(tic)
	print('time used: ',toc)

	if t < 10 then checkMemory() end
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
  convForward_4:remember('eval') 
  convForward_4:evaluate()
  print(c.blue '==>'..'validating')
  local t = 0
while t < 2037 do
	t = t + 1
  	if testFlag and t == 2 then 
  		print('testing mode')
  		testFlag = false
  		t = 2000 - 1
  		print('exit testing mode')
  	end

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
    local  inputTable = {}

	for i = 1, opt.input_nSeq do
  	    if gpuflag then
            table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
        else
        	table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
        end
	end


	output0 = encoder_0:updateOutput(inputTable)
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

	decoder_3:updateOutput(decoder_2.output)
	output3 = decoder_3.modules[1].outputs

--	inputTable4 = {{{output0[opt.input_nSeq]}, output2},{{output1[opt.input_nSeq]}, output3}}
	inputTable4 = {output2, output3}

	output = convForward_4:forward(inputTable4)

    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	if wrapOpticalFlow then   
            local optical_flow = convForward_4.modules[3].modules[1].modules[1].modules[1].modules[2].modules[3].modules[7].output
    	    local imflow = flow2colour(optical_flow)
            image.save(epochSaveDir..'flow-iter'..tostring(iter)..'.png', imflow)
        end
        
        saveImage(output, 'valid-output', iter, epochSaveDir, 'output')
        saveImage(output2, 'valid-output2', iter, epochSaveDir, 'output')
        saveImage(output3, 'valid-output3', iter, epochSaveDir, 'output')
        saveImage(output0, 'valid-output0', iter, epochSaveDir, 'output')
        print('image save')
    end

	criterion = nn.SequencerCriterion(nn.MSECriterion()):float()
	if gpuflag then criterion:cuda() end
     
   local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW):float()
   local target = {}
--	local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.nFiltersMemory[1], opt.width, opt.width):float()

	if opt.batchSize == 1 then
		for i = 1, opt.output_nSeq do
		    if gpuflag then 
		        target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i]):cuda()
		    else
		    	target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i])
		    end
		end
	else
		print('opt.batchSize > 1 !!')
		target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:select(2,1):resizeAs(targetSeq[i])
	end


    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	saveImage(target, 'target', iter, epochSaveDir, 'output')
	end

	err = criterion:forward(output, target)
    print('valid score:',err)
	totalerror = err + totalerror
	
end
    print("\tvalidation total is ", totalerror)
end

--- =============================
--- ==============================

for k = 2, 200 do -- max epoch = 299

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
		
    	if wrapOpticalFlow then   
    		print('test optical flow')
            local optical_flow = convForward_4.modules[3].modules[1].modules[1].modules[1].modules[2].modules[3].modules[7].output
		end

		print('test train ')
		checkMemory()
		testFlag = true
		train() 

		print('test valid')
		checkMemory()
		testFlag = true
		valid()

	   	print('test model save, move to CPU')
	   	checkMemory()
	--        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
        encoder_0:float()
		encoder_1:float() 
		decoder_2:float()
		decoder_3:float() 
		convForward_4:float() 

		checkMemory()
	   	saveModel(encoder_0, 'test_encoder_0', k)
	   	saveModel(encoder_1, 'test_encoder_1', k)
	   	saveModel(decoder_2, 'test_decoder_2', k)
	   	saveModel(decoder_3, 'test_decoder_3', k)
	   	saveModel(convForward_4, 'test_convForward_4', k)
	   	print('done modelSave')
	   	checkMemory()
		print('move to GPU')
		if gpuflag then
	        encoder_0:cuda()
			encoder_1:cuda() 
			decoder_2:cuda()
			decoder_3:cuda() 
			convForward_4:cuda() 
		end
		print('run all valid first')
		valid() -- run all valid first
		print(c.red 'pass test!')
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
		
		checkMemory()
		print('move to GPU')
                encoder_0:cuda()
		encoder_1:cuda() 
		decoder_2:cuda()
		decoder_3:cuda() 
		convForward_4:cuda() 
	end

end
