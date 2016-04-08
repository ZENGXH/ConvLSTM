-- warp with optical flow
print("info: different from test_completeWrap_clamp, this one do not use mask to subsample, but use image scale, since using mask causing confusion during BilinearSamplerBHWD")
require 'torch'
dofile('./hko/opts-hko.lua')    
---- always load opts first!

unpack = unpack or table.unpack
onMac = opt.onMac
saveOutput = true
resumeFlag = false
resumeIter = 1
verbose = true
display = false

local saveInterval = 10
local learningRate = 1e-4

require 'paths'
modelDir = 'image/completeWrap_subsample/'
imageDir = 'image/completeWrap_subsample/'
if not paths.dirp(modelDir) then os.execute('mkdir -p ' .. modelDir) end
if not paths.dirp(imageDir) then os.execute('mkdir -p ' .. imageDir) end
print('image dir: ',imageDir)
print('model dir:', modelDir)

gpuflag = opt.gpuflag
wrapOpticalFlow = true
underLowMemory = false

local c = require 'trepl.colorize'

require 'torch_extra/modelSave'
require 'nn'

require 'image'
require 'optim'
require 'ConvLSTM'
require 'display_flow'
require 'DenseTransformer2D'
require 'rnn'
require 'ConvLSTM_NoInput'
require 'stn'
require 'torch_extra/imageScale'
dofile 'torch_extra/dataProvider.lua'
dofile 'flow.lua'
dofile 'torch_extra/SelfFeedSequencer.lua'

if not onMac then
	require 'cunn'
	require 'cutorch'
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

print(opt)
if not opt.gpuflag then
	torch.setdefaulttensortype('torch.FloatTensor')
	typeT = torch.Tensor():float()
	print("setdefaulttensortype: ", typeT)
else
	typeT = torch.Tensor():cuda()
	print("setdefaulttensortype: ", typeT)
end
local std = opt.parametersInitStd 


function checkMemory(message) 
  	if not onMac then
  		print(message)
		local freeMemory, totalMemory = cutorch.getMemoryUsage(gpuid)
		print('free',freeMemory/(1024^3))
	end
end


function saveImage(figure_in, name, iter, epochSaveDir, type, numsOfOut)
	local numsOfOut = numsOfOut or 0
	local figure = torch.Tensor()
	if torch.isTensor(figure_in)  then 
    	local img = figure_in:clone():typeAs(typeT)
    	img = img:mul(1/img:max()):squeeze()
	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img)
    elseif type == 'output' then------------- table --------------	
    	for numsOfOut = 1, table.getn(figure_in) do
    		------- choose the first batch along 
    		local img = figure_in[numsOfOut][1]:clone()  --- 
    		saveImage(img, name, iter, epochSaveDir, type, numsOfOut)
    	end
    else 
     	for numsOfOut = 1, table.getn(figure_in) do
    		------- choose the first batch along 
    		local img = figure_in[numsOfOut]:clone()  --- 
    		saveImage(img, name, iter, epochSaveDir, type, numsOfOut)
    	end   	
	end
end

if not onMac then
	data_path = '/csproject/dygroup2/xiaohui/ConvLSTM/helper/'
else
	data_path = '../ConvLSTM/helper/'	
end

trainDataProvider = getdataSeq_hko('train', data_path)
validDataProvider = getdataSeq_hko('valid', data_path)

-- dataLoad = {intputTable, outputTable}
-- darasetSeq_valid = getdataSeq_hko('valid', data_path)

print('==> training model')
checkMemory()
torch.manualSeed(opt.seed)  

------- ================
local eta0 = 1e-6
local eta = opt.eta
local errs= 0
local iter = 0
local epoch = 0

if gpuflag then print('gpu flag on') else print('gpu flag off') end

iter = 0

scaleDown = nn.imageScale(opt.inputSizeH, opt.inputSizeW, 'bicubic')
scaleUp = nn.imageScale(opt.imageH, opt.imageW, 'bicubic')-- :typeAs(typeT)
if(gpuflag) then
	scaleUp:cuda()
	scaleDown:cuda()
end

if resumeFlag then
	print('==> load model, resume traning')
  	encoder_0 = torch.load(modelDir..'encoder_0iter-'..resumeIter..'.bin')
  	repeatModel = torch.load(modelDir..'repeatModeliter-'..resumeIter..'.bin')
  	flowGridGenerator = repeatModel.modules[1].modules[1].modules[1].modules[1].modules[2].modules[2].modules[1]
  	print(repeatModel)
  	print('load model done')
 	local lstm_params4, lstm_grads4 = repeatModel:getParameters()
	-- lstm_params4:normal(opt.paraInit, std)
	local lstm_params0, lstm_grads0 = encoder_0:getParameters()
	print('number of parameters of repeatModel', lstm_params4:size(1) + lstm_params0:size(1))
	-- 91552
else

	print('==> build model for traning')
--------------------------------------  building model -----------------------------


	local baseLSTM = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq - 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, true, 3):float()
	encoder_0 = nn.Sequencer(baseLSTM):float()-- batchsize

	local lstm_params0, lstm_grads0 = encoder_0:getParameters()
	lstm_params0:uniform(-0.08, 0.08)

    local reshape = nn.View(opt.batchSize, opt.fakeDepth, opt.memorySizeH, opt.memorySizeW )

	local predictBranch = nn.Sequential() -- do convolution to 1*17*50*50 input , output size == itarget
	                               :add(nn.View(opt.batchSize, opt.nFiltersMemory[2], opt.width, opt.width ))
	                               :add(nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
							       :add(nn.View(opt.batchSize, opt.nFiltersMemory[1], opt.memorySizeH, opt.memorySizeW )) -- 1, 4, 50, 50 => 1 50 50 4
							       --:add(nn.Transpose({2,4}, {2,3})):typeAs(typeT) -- batchSie, H, W, fakeDepth
							       -- output b1 h50 w50 d4: 
							       --:add(nn.Transpose({1,4})) -- make fake Depth batchSIZE
							       -- output 4 50 50 1

	flowGridGenerator = nn.Sequential():add(flow) 
	print(flowGridGenerator)   					
		    						-- B1 D4 H50 W50 => B4 D1 H50 W50
		-- for input of flow: B4 D1 H50 W50, 
		-- depth of first conv should be opt.batchSize -> others
	    							
	    							-- output 4 50 50 2

	local inputBranch = nn.Sequential()
							:add(nn.Transpose({4,2},{2,3})) -- b 17 50 50 bdhw TO BE BHWD

	flowWrapper = nn.Sequential()
							:add(nn.NarrowTable(1, 2))
							--:add(nn.ConcatTable():add(flowGridGenerator):add(input_tranpose))
							:add(nn.BilinearSamplerBHWD()) -- output b h w d: 4 50/h 50/w 1
		    		        :add(nn.Transpose({3,4}, {2,3})) -- bhwd -> bdhw 4(bathSize which is fake depth) 1 50/h 50/w
					        -- :add(nn.Transpose({1, 2})) -- return to 4 as depth
	-- expected output: b h w d	
	local interface = nn.Sequential()
							:add(nn.FlattenTable()) -- {prediction, flowGrid, inputImage}
							:add(nn.ConcatTable():add(nn.SelectTable(3)):add(flowWrapper))
    baseLSTMCopy = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq - 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, true, 3):float()
    baseLSTMCopy:share(baseLSTM, 'weight','bias')

    local lstmBranch = nn.Sequential()
    						:add(baseLSTMCopy)
    						:add(nn.ConcatTable():add(flowGridGenerator):add(predictBranch))

	local outerConcat = nn.Sequential()
						:add(nn.ConcatTable():add(inputBranch):add(lstmBranch))
						:add(interface) -- {nn.flatten, nn.}

	local mainBranch = nn.Sequential()
						:add(outerConcat)
						:add(nn.CAddTable())
						:add(nn.Clamp(0.00,1.00)):float()
    repeatModel = nn.SelfFeedSequencer(mainBranch):float()
	local lstm_params4, lstm_grads4 = repeatModel:getParameters()
	lstm_params4:normal(opt.paraInit, std)

	print('number of parameters of repeatModel', lstm_params4:size(1) + lstm_params0:size(1))
	----------------------------------------------------------------------------
end


print('\nencoder_0')
print(encoder_0)

print('\nmainBranch')
print(mainBranch)

print("\nouterConcat")
print(outerConcat)
-- print(outerConcat.modules)
----- *********************************************************

--[[


print('lstmBranch: \n', lstmBranch.output) -- {4 50 50 1 , 4 50 50 2 }

print('predictBranch: \n', predictBranch.output:size())  -- 1 4, 50 50 
print('flowGridGenerator: \n', flowGridGenerator.output:size()) -- 4 50 50 2
print('inputBranch: \n', inputBranch.output:size()) -- 4 50 50 1

print('flowWrapper: \n', flowWrapper.output:size())  -- 1 4 50 50
print('interface: \n', interface.output)  -- table

print('outerConcat: \n', outerConcat.output:size()) -- 1 4 50 50 
print('mainBranch: \n', mainBranch.output:size())

]]
----- **********************************************************
function train()
	epoch = epoch or 1  
	local epochSaveDir = imageDir..'train-epoch'..tostring(epoch)..'/'
 	if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
  
	--encoder_0:remember('eval')
	--encoder_0:evaluate()
	encoder_0:remember('both')
	encoder_0:training()

	repeatModel:remember('both') 
	repeatModel:training()
	local accErr = 0
	for t =1, opt.trainIter do
	    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	    local tic = torch.tic()

	    local iter = t
	    local trainData = trainDataProvider[t]

	    encoder_0:zeroGradParameters()
	    encoder_0:forget()

	    repeatModel:zeroGradParameters()
	    repeatModel:forget()
	    local inputTable = trainData[1]
	    -- assert(baseLSTMCopy.step == 1)
	    --------------************--------------------------------------hard code input-----------
	    
		-- rescale for training
	    --print('before scaling: ', inputTable)
	    inputTable = scaleDown:forward(inputTable)
	    local encoderInput = {}

	    if opt.gpuflag then 
	    	for i = 1, table.getn(inputTable) - 1 do
	    		encoderInput[i] = inputTable[i]:cuda()
	    	end
	    	encoder_0:cuda()
	    else 
	    	for i = 1, table.getn(inputTable) - 1 do
	    		encoderInput[i] = inputTable[i]
	    	end
	    end
	    -- print('inputTable', inputTable)
	    -- print('encoder_0 input', encoderInput)

	    local inputTable4 = {}
	    
	    if opt.gpuflag then inputTable4[1] = inputTable[opt.input_nSeq]:cuda()
	    else inputTable4[1] = inputTable[opt.input_nSeq]
	    end
	    inputTable = {}
	    if opt.gpuflag then
		    for i = 1, opt.output_nSeq - 1 do -- fill in dummpy input
			    inputTable4[i + 1] = torch.Tensor():cuda():resizeAs(inputTable4[1]):fill(0)
			end
		else
		    for i = 1, opt.output_nSeq - 1 do -- fill in dummpy input
			    inputTable4[i + 1] = torch.Tensor():resizeAs(inputTable4[1]):fill(0)
			end
		end
		--------------************-------------------------------------------------
		local output0 = encoder_0:forward(encoderInput)
		-- print('encoder_0 output', output0)
	    -- print('output of encoder_0: ', output0)
		output = repeatModel:forward(inputTable4)
--		print("========================")	
	    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
	    	if wrapOpticalFlow then  
	    		local optical_flow = flowGridGenerator.modules[1].modules[5].output -- 4 2 50 50
    			local flow_batch1 = optical_flow[1]
 		   	    local imflow = flow2colour(flow_batch1)
            	saveImage(imflow, 'flow', iter, epochSaveDir, 'flow', i)		       
	        end
	        local lastinput = repeatModel.modules[1].modules[1].modules[1].modules[1].modules[1].output
	        -- print(lastinput:size())
	        saveImage(lastinput[1]:select(3,1), 'lastinput', iter, epochSaveDir, 'output', i)	
	       -- saveImage(output0, 'output0', iter, epochSaveDir, 'output')
	    end

		--criterion = nn.MSECriterion():typeAs(typeT)
		local criterion = nn.SequencerCriterion(nn.MSECriterion(1))
		if opt.gpuflag then criterion:cuda() end
	     
		local target = trainData[2]
	    -- local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.fakeDepth, opt.inputSizeH, opt.inputSizeW):typeAs(typeT)

		--print('criterion input: ')
		--print('output', output)
		--print('target', target)
		--print('inputTable',inputTable)

	    if saveOutput and math.fmod(t , saveInterval) == 1 or t == 1 then
	    	saveImage(output, 'output', iter, epochSaveDir, 'output')
	    	saveImage(target, 'target', iter, epochSaveDir, 'output')
	    	-- { Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), }
		end
--lstm = repeatModel.modules[1].modules[1].modules[1].modules[1].modules[2].modules[1].modules[1]
-- flow = repeatModel.modules[1].modules[1].modules[1].modules[1].modules[2].modules[2].modules[1]

		target = scaleDown:forward(target)
	    if opt.gpuflag then 
	    	for i = 1, table.getn(target) do
	    		target[i] = target[i]:cuda()
	    	end
	    end

		err = criterion:forward(output, target)
		accErr = err + accErr
		print("\titer",t, "err:", err)

		if t < 10 then checkMemory('\ncriterion start bp <<<<') end

		local gradOutput = criterion:backward(output, target)

	    if saveOutput and math.fmod(t , saveInterval) == 1 or t == 1 then
	    	saveImage(output, 'output', iter, epochSaveDir, 'output')
	    	saveImage(target, 'target', iter, epochSaveDir, 'output')
	    	-- { Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), }
		end
		if t < 10 then checkMemory('criterion backward done <<<<') end
		--------------************-------------
		for i = 1, #inputTable4 - 1 do
			inputTable4[i] = output[i + 1]
		end	
		--------------************-------------
	    target = {}
		output = {}
		--print(gradOutput)
		--checkMemory()
	-------------------------------
	    --local a, b = flowGridGenerator:getParameters()
	    --print('para mean of flowGridGenerator', a:mean())
	    --print('gradpara mean of flowGridGenerator', b:mean())		
		if t < 10 then checkMemory('\nconv_4 bp <<<<') end

		-- lstm_param1, lstm_grad1 = baseLSTMCopy:getParameters()
		-- gradOutput[1] = gradOutput[1]:mul(1000)
		-- gradOutput[2] = gradOutput[2]:mul(10)

		repeatModel:backward(inputTable4, gradOutput)

	    --repeatModel:backwardUpdate(inputTable4, gradOutput, learningRate)
--      repeatModel:updateGradInput(inputTable4, gradOutput)  
--		repeatModel:accGradParameters(inputTable4, gradOutput, 1)
		-- conv0 = flowGridGenerator.modules[1].modules[1].modules[1].gradBias:fill(1)
		-- print('bias', flowGridGenerator.modules[1].modules[1].modules[1].bias)
--		print('mean', flowGridGenerator.modules[1].modules[1].modules[1].weight:mean())
		repeatModel:updateParameters(learningRate)

	    --print('gradoutput', gradOutput[1]:mean())
	    --local a, b = lstm:getParameters()
	    --print('para mean of flowGridGenerator', a:mean())
	    --print('gradpara mean of flowGridGenerator', b:mean())

	    repeatModel:zeroGradParameters()
	    repeatModel.output = {}

	--[[
	    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
	    	for i = 1, 4 do
		    	quicksaveImage(gradOutput:select(2, i), 'gradOutput', iter, epochSaveDir, 'grad', i)
		    end
	        saveImage(gradOutput, 'gradOutput', iter, epochSaveDir, 'grad')
		end
		gradOutput = {}
	]]
		print('\nconv_4 bp done')  
		-- lstm_param2, lstm_grad2 = baseLSTMCopy:getParameters()
		--lstm_param2, lstm_grad = baseLSTM:getParameters()
		-- assert(lstm_param1:eq(lstm_param2):max() == 0)

		inputTable =  {}
		repeatModel:forget()
		if math.fmod(t, 80)  == 1 then
			print('accE rror of ', 80 * opt.batchSize, 'is >>>>>> ', accErr)
			accErr = 0
		end
	--	print('encoder_0 bp done:')
		if t < 10 then checkMemory("backward done") end
		local toc = torch.toc(tic)
		print('time used: ',toc)

		if t < 10 then checkMemory() end
		if opt.onMac and t == opt.trainIter / 2 then
			saveModel(modelDir, encoder_0, 'encoder_0_middle', epoch)
		   	saveModel(modelDir, repeatModel, 'repeatModel_middle', epoch)
	    end
	end

	epoch = epoch + 1

end


-- **************************************
function valid()
	local totalerror = 0
	local epochSaveDir = imageDir..'valid/'
	if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
	encoder_0:remember('eval')
	encoder_0:evaluate()

	repeatModel:remember('both') 
	repeatModel:training()
	print(c.blue '==>'..'validating')
	local t = 0

	while t < opt.validIter do
		t = t + 1
	  	if testFlag and t == 2 then 
	  		print('testing mode')
	  		testFlag = false
	  		t = opt.validIter - 3 - 1
	  		print('exit testing mode')
	  	end

	  	local validData = validDataProvider[t]
	    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	    local tic = torch.tic()
	    local iter = t

	    encoder_0:zeroGradParameters()
	    encoder_0:forget()

	    repeatModel:zeroGradParameters()
	    repeatModel:forget()

	    local inputTable = validData[1]
	    --------------************--------------------------------------hard code input-----------
	    
		-- rescale for training
	    --print('before scaling: ', inputTable)
	    inputTable = scaleDown:forward(inputTable)
	    local encoderInput = {}

	    if opt.gpuflag then 
	    	for i = 1, table.getn(inputTable) - 1 do
	    		encoderInput[i] = inputTable[i]:cuda()
	    	end
	    	encoder_0:cuda()
	    else 
	    	for i = 1, table.getn(inputTable) - 1 do
	    		encoderInput[i] = inputTable[i]
	    	end
	    end
	    -- print('inputTable', inputTable)
	    -- print('encoder_0 input', encoderInput)

	    local inputTable4 = {}
	    
	    if opt.gpuflag then inputTable4[1] = inputTable[opt.input_nSeq]:cuda()
	    else inputTable4[1] = inputTable[opt.input_nSeq]
	    end
	    inputTable = {}
	    if opt.gpuflag then
		    for i = 1, opt.output_nSeq - 1 do -- fill in dummpy input
			    inputTable4[i + 1] = torch.Tensor():cuda():resizeAs(inputTable4[1]):fill(0)
			end
		else
		    for i = 1, opt.output_nSeq - 1 do -- fill in dummpy input
			    inputTable4[i + 1] = torch.Tensor():resizeAs(inputTable4[1]):fill(0)
			end
		end

		--------------************-------------------------------------------------
		local output0 = encoder_0:forward(inputTable)
	    
		output = repeatModel:forward(inputTable4)

		local criterion = nn.SequencerCriterion(nn.MSECriterion())
		--criterion = nn.MSECriterion():typeAs(typeT)
		if gpuflag then criterion:cuda() end
	     
		local target = validData[2]
		target = scaleDown:forward(target)

	    if opt.gpuflag then 
	    	for i = 1, table.getn(target) do
	    		target[i] = target[i]:cuda()
	    	end
	    end

		local err = criterion:forward(output, target)
		target = {}
		output = {}

		print("\titer", t, "valid score:", err)	
		err = criterion:forward(output, target)

		totalerror = err + totalerror
	end
    print("\tvalidation total is ", totalerror)
end

--- =============================
--- ==============================

for k = 1, opt.maxEpoch do -- max epoch = 299

	if k == opt.maxEpoch * 2/3 then
		learningRate = learningRate * 1e-1
		print("learningRate reduce, " ,learningRate)
	end


	if opt.gpuflag then 
		checkMemory('before move to GPU')
        encoder_0:cuda()
		repeatModel:cuda() 
		checkMemory('after: ')
	end 
		print('test optical flow')
	   	local optical_flow = flowGridGenerator.modules[1].modules[5].output -- 4 2 50 50
		testFlag = false
	if k == 1 then 
		
    	if wrapOpticalFlow then   
    		print('test optical flow')
    		local optical_flow = flowGridGenerator.modules[1].modules[5].output
		end

		print('test train ')
		checkMemory("")
		testFlag = true
		valid()	
	   	saveModel(modelDir, encoder_0, 'test_encoder_0', k)
	   	saveModel(modelDir,repeatModel, 'test_repeatModel', k)	
	   	print("save model pass test")
		train() 

		print('test valid')
		checkMemory("")
		testFlag = true


	   	print('test model save, move to CPU')
	   	checkMemory("")
	--        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
        encoder_0:float()
		repeatModel:float()

		checkMemory()
	   	saveModel(modelDir, encoder_0, 'test_encoder_0', k)
	   	saveModel(modelDir,repeatModel, 'test_repeatModel', k)
	   	print('done modelSave')
		if gpuflag then
			checkMemory('move to GPU')
	        encoder_0:cuda()
			repeatModel:cuda() 
		end
		print('run all valid first')
		-- valid() -- run all valid first
		print(c.red 'pass test!')
	end

		train()
		valid()

	if gpuflag then 
		encoder_0:float()
		repeatModel:float()
		if math.fmod(k , 1) == 1 then
	        saveModel(modelDir, encoder_0, 'encoder_0', k)
		   	saveModel(modelDir, repeatModel, 'repeatModel', k)
	    end
	   	print('modelSave')
		
		checkMemory('move to GPU')
        encoder_0:cuda()
		repeatModel:cuda() 
	end

end
