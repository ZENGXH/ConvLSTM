-- warp with optical flow
print('info: use mask to perform subsampling, 4 as the depth of the input image, 
	but it become confuse with the batch Size, since when perfoming the BilinearSamplerBHWD, it is 
	require that the depth of the inputImage is 1. decide to the maxpooling or other way of subsampling to 
	accelate the training, increase the batchSize, see test_subsample for further infomation')
unpack = unpack or table.unpack
onMac = true
saveOutput = true
resumeFlag = false
resumeIter = 1
verbose = true
display = false
require 'paths'
local modelDir = 'image/completeWrap/'
local imageDir = 'image/completeWrap/'

if not paths.dirp(imageDir) then
	os.execute('mkdir -p ' .. imageDir)
end
print('image dir: ',imageDir)

if not paths.dirp(modelDir) then
	os.execute('mkdir -p ' .. modelDir)
end
print('model dir:', modelDir)

if onMac then
	gpuflag = false
	wrapOpticalFlow = true
else
	gpuflag = true
	wrapOpticalFlow = true
end

underLowMemory = false

local c = require 'trepl.colorize'

require 'torch_extra/modelSave'
require 'nn'
require 'torch'

local saveInterval = 10
local learningRate = 1e-6

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
dofile('flow.lua') 
require  'stn'
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


function saveImage(figure_in, name, iter, epochSaveDir, type,numsOfOut)
	local numsOfOut = numsOfOut or 0
	local figure = torch.Tensor()
	if torch.isTensor(figure_in)  then 
		if figure_in:size(2) == 4 then 
--			print(figure_in:size())
            img = dataMask:quick4Mask3Dbackward(figure_in):float() -- :resize(1,100,100)
            quicksaveImage(img, 'all-'..name, iter, epochSaveDir, type, numsOfOut)
        else
        	quicksaveImage(figure_in, name, iter, epochSaveDir, type, numsOfOut)
        end
    else		
    	for numsOfOut = 1, table.getn(figure_in) do
    		local img = figure_in[numsOfOut]:clone()
    		saveImage(img, name, iter, epochSaveDir, type,numsOfOut)
    	end
	end
end
--[[
	if not torch.isTensor(figure_in)  then 
		if table.getn(figure_in) == 1 then
			figure = figure_in[1]:clone()
		end
    else
    	figure = figure_in:clone()
	end

	if torch.isTensor(figure) and type == 'flow' then
		local img = figure:clone():float() -- :resize(1, 100, 100)
   	        if type == 'flow' and display then
			    print('min and max of '..name..' is', img:min(),img:max())
		    	print(img[1]:mean(),img[2]:mean(),img[3]:mean())
   	        	image.display(img)
   	        end		
	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-.png',  img/img:max())
	elseif type == 'output' then
		if torch.isTensor(figure) then
	    	local img = torch.Tensor()
    		if figure:size(2) == 4 then 
   	            img = dataMask:quick4Mask3Dbackward(figure):float() -- :resize(1,100,100)
   	        else 
   	        	img = figure:clone()
            end
   	        if name == 'output' and display then
		        print('min and max of '..name..' is', img:min(),img:max(), img:mean())
   	        	image.display(img)
   	        end    	    
    	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
    	else 
    		for numsOfOut = 1, 
	elseif type == 'gard' then 
		for numsOfOut = 1, table.getn(figure) do
   	    	local img = dataMask:quick4Mask3Dbackward(figure):float():resize(1, opt.nFiltersMemory[2] * 100, 100)
   	    	img = img:mul(1/img:max())
    	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
    	end  
	end
	]]


function quicksaveImage(figure_in, name, iter, epochSaveDir, type, numsOfOut)
	local numsOfOut = numsOfOut or 0
	if type == 'output' then
    	local img = figure_in:clone():float()
    	-- img = img:mul(1/img:max())
	    image.save(epochSaveDir..'quick-iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img)
	elseif type == 'grad' then
    	local img = figure_in:clone():float()
    	-- img = img:mul(1/img:max())
	    image.save(epochSaveDir..'quick-iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
	elseif type == 'flow' then
		local img = figure_in:clone():float()
	    image.save(epochSaveDir..'quick-iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
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



if gpuflag then print('gpu flag on') else print('gpu flag off') end

local iter = 0

if resumeFlag then
	print('==> load model, resume traning')
  	encoder_0 = torch.load(modelDir..'test_encoder_0iter-'..resumeIter..'.bin')
  	mainBranch = torch.load(modelDir..'test_mainBranchiter-'..resumeIter..'.bin')
  	print('load model done')
else
	print('==> build model for traning')
--------------------------------------  building model -----------------------------
	baseLSTM = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq - 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, true, 3):float()
	encoder_0 = nn.Sequencer(baseLSTM):float()-- batchsize

	local lstm_params0, lstm_grads0 = encoder_0:getParameters()
	lstm_params0:uniform(-0.08, 0.08)

    local reshape = nn.View(opt.batchSize, opt.fakeDepth, opt.memorySizeH, opt.memorySizeW )

	predictBranch = nn.Sequential() -- do convolution to 1*17*50*50 input , output size == itarget
	                               :add(nn.View(opt.batchSize, opt.nFiltersMemory[2], opt.width, opt.width ))
	                               :add(nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
							       :add(nn.View(opt.batchSize, opt.fakeDepth, opt.memorySizeH, opt.memorySizeW )) -- 1, 4, 50, 50 => 1 50 50 4
							       --:add(nn.Transpose({2,4}, {2,3})):float() -- batchSie, H, W, fakeDepth
							       -- output b1 h50 w50 d4: 
							       --:add(nn.Transpose({1,4})) -- make fake Depth batchSIZE
							       -- output 4 50 50 1

	flowGridGenerator = nn.Sequential()
									:add(nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
	    							:add(nn.Transpose({1,2})) -- make fakeDepth batch 
		    						-- B1 D4 H50 W50 => B4 D1 H50 W50
		-- for input of flow: B4 D1 H50 W50, 
		-- depth of first conv should be opt.batchSize -> others
	    							:add(flow) 
	    							-- output 4 50 50 2

	inputBranch = nn.Transpose({1,2},{2,3},{3,4}) -- 17 50 50 1

	flowWrapper = nn.Sequential()
							:add(nn.NarrowTable(1, 2))
							--:add(nn.ConcatTable():add(flowGridGenerator):add(input_tranpose))
							:add(nn.BilinearSamplerBHWD()) -- output b h w d: 4 50/h 50/w 1
		    		        :add(nn.Transpose({3,4}, {2,3})) -- bdhw 4(bathSize which is fake depth) 1 50/h 50/w
					        :add(nn.Transpose({1, 2})) -- return to 4 as depth
	-- expected output: b h w d	
	interface = nn.Sequential()
							:add(nn.FlattenTable()) -- {prediction, flowGrid, inputImage}
							:add(nn.ConcatTable():add(nn.SelectTable(3)):add(flowWrapper))
    baseLSTMCopy = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq - 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, true, 3):float()
    baseLSTMCopy:share(baseLSTM, 'weight','bias')

    lstmBranch = nn.Sequential()
    						:add(baseLSTMCopy)
    						:add(nn.ConcatTable():add(flowGridGenerator):add(predictBranch))

	outerConcat = nn.Sequential()
						:add(nn.ConcatTable():add(inputBranch):add(lstmBranch))
						:add(interface) -- {nn.flatten, nn.}

	mainBranch = nn.Sequential()
						:add(outerConcat)
						:add(nn.CAddTable())
						:add(nn.Clamp(0.00,1.00)):float()
    repeatModel = nn.Repeater(mainBranch, opt.output_nSeq):float()
	local lstm_params4, lstm_grads4 = repeatModel:getParameters()
	lstm_params4:normal(paraInit, std)
	local p, g = repeatModel:getParameters()
	print('number of parameters of repeatModel', p:size())
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

  if not paths.dirp(epochSaveDir) then
   os.execute('mkdir -p ' .. epochSaveDir)
  end
  
  encoder_0:remember('both')
  encoder_0:training()

  repeatModel:remember('both') 
  repeatModel:training()
  --t = 0
  local shuffleInd = torch.randperm(2037 * 4)
  for t =1, 2037 * 4 do
  	--t = t + 1
 --[[
  	if testFlag and t == 2 then 
  		print('testing mode')
  		print('exit testing mode')
  		t = 2000 * 4 - 1
  	end]]

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local tic = torch.tic()

    iter = t

    encoder_0:zeroGradParameters()
    encoder_0:forget()

    repeatModel:zeroGradParameters()
    repeatModel:forget()

    local sample = datasetSeq[shuffleInd[t]] 
    local data = sample[1]:float()

	data:resize(opt.batchSize, opt.nSeq, opt.fakeDepth, opt.inputSizeH, opt.inputSizeW):float()

    local  inputTable = {}

	for i = 1, opt.input_nSeq do
		if gpuflag then
	        table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
	    else
	    	table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
	    end
	end
	output0 = encoder_0:updateOutput(inputTable)
    
    inputTable4 = inputTable[opt.input_nSeq]
    -- print('output of encoder_0: ', output0)

	output = repeatModel:forward(inputTable4)


    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	if wrapOpticalFlow then  
    		local optical_flow = flowGridGenerator.modules[3].modules[5].output -- 4 2 50 50
 --   		print('optical flow is 4 * nDimension', optical_flow[1]:nDimension())
    		for i = 1, 4 do
    			local flow_depth = optical_flow[i]
 		   	    local imflow = flow2colour(flow_depth)
            	saveImage(imflow, 'flow-n'..i, iter, epochSaveDir, 'flow')
	        end
        end
        
       -- saveImage(output0, 'output0', iter, epochSaveDir, 'output')
    end

	criterion = nn.SequencerCriterion(nn.MSECriterion()):float()
	--criterion = nn.MSECriterion():float()
	if gpuflag then criterion:cuda() end
     
    local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.fakeDepth, opt.inputSizeH, opt.inputSizeW):float()
    local target = {}
--	local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.nFiltersMemory[1], opt.width, opt.width):float()
	if opt.batchSize == 1 then
		for i = 1, opt.output_nSeq do
            if gpuflag then
		        target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i]):cuda()
		    else
                target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i]):float()
            end
		end

	else
		print('opt.batchSize > 1 !!')
--		targetSeq[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:select(2,1):resizeAs(targetSeq[i])
	end

	print('criterion')
--	print(output)
--	print(target)

	if opt.output_nSeq == 1 then
		target = target[1]
	end
 if gpuflag then targetSeq = targetSeq:cuda() end

    if saveOutput and math.fmod(t , saveInterval) == 1 or t == 1 then
    	saveImage(output, 'output', iter, epochSaveDir, 'output')
    	for i = 1, 4 do
	    	quicksaveImage(output[2]:select(2, i), 'output[2]', iter, epochSaveDir, 'output', i)
	    end

    	saveImage(target, 'target', iter, epochSaveDir, 'output')
--    	print('size of target[1]:',target[1]:size())
    	for i = 1, 4 do
	    	quicksaveImage(target[2]:select(2, i), 'target[2]', iter, epochSaveDir, 'output', i)
	    end    	
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
	mainBranch.outputs = {}

--	print(gradOutput)
	--checkMemory()

-------------------------------
	
	if t < 10 then 
		checkMemory() 
		print('\nconv_4 bp <<<<')  -- 1
	end

    repeatModel:backwardUpdate(inputTable4, gradOutput, learningRate)
    
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
lstm_param, lstm_grad = baseLSTMCopy:getParameters()

lstm_param2, lstm_grad = baseLSTM:getParameters()

assert(lstm_param:eq(lstm_param2):max() == 0)

-- print('gardOutput_encoder_0', lstm_param:size(), lstm_param2:size())
--	encoder_0:backwardUpdate({inputTable[opt.input_nSeq-1]}, {gardOutput_encoder_0}, learningRate)
--	encoder_0.module:forget()
--	encoder_0:zeroGradParameters()


	inputTable =  {}
	
	repeatModel:forget()
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
 
  mainBranch:remember('eval') 
  mainBranch:evaluate()
  print(c.blue '==>'..'validating')
  local t = 0

while t < 2037 do
	t = t + 1
  	if testFlag and t == 2 then 
  		print('testing mode')
  		testFlag = false
  		t = 2020 - 1
  		print('exit testing mode')
  	end

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local tic = torch.tic()

    iter = t

    encoder_0:zeroGradParameters()
    encoder_0:forget()

    repeatModel:zeroGradParameters()
    repeatModel:forget()

    local sample = datasetSeq[t] 
    local data = sample[1]:float()

	data:resize(opt.batchSize, opt.nSeq, opt.fakeDepth, opt.inputSizeH, opt.inputSizeW):float()

    local  inputTable = {}

	for i = 1, opt.input_nSeq do
		if gpuflag then
	        table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width):cuda())
	    else
	    	table.insert(inputTable, data[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(1, opt.nFiltersMemory[1], opt.width, opt.width))
	    end
	end
	output0 = encoder_0:updateOutput(inputTable)
    
    inputTable4 = inputTable[opt.input_nSeq]
    -- print('output of encoder_0: ', output0)

	output = repeatModel:forward(inputTable4)


    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	if wrapOpticalFlow then  
    		local optical_flow = flowGridGenerator.modules[3].modules[5].output -- 4 2 50 50
 --   		print('optical flow is 4 * nDimension', optical_flow[1]:nDimension())
    		for i = 1, 4 do
    			local flow_depth = optical_flow[i]
 		   	    local imflow = flow2colour(flow_depth)
            	saveImage(imflow, 'flow-n'..i, iter, epochSaveDir, 'flow')
	        end
        end
        
       -- saveImage(output0, 'output0', iter, epochSaveDir, 'output')
    end

	criterion = nn.SequencerCriterion(nn.MSECriterion()):float()
	--criterion = nn.MSECriterion():float()
	if gpuflag then criterion:cuda() end
     
    local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.fakeDepth, opt.inputSizeH, opt.inputSizeW):float()
    local target = {}
--	local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.nFiltersMemory[1], opt.width, opt.width):float()
	if opt.batchSize == 1 then
		for i = 1, opt.output_nSeq do
            if gpuflag then
		        target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i]):cuda()
		    else
                target[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:resizeAs(targetSeq[i]):float()
            end
		end

	else
		print('opt.batchSize > 1 !!')
--		targetSeq[i] = data[{{}, {opt.input_nSeq + i}, {}, {}, {}}]:select(2,1):resizeAs(targetSeq[i])
	end

--	print('criterion')
--	print(output)
--	print(target)

	if opt.output_nSeq == 1 then
		target = target[1]
	end
 if gpuflag then targetSeq = targetSeq:cuda() end

    if saveOutput and math.fmod(t , saveInterval) == 1 or t == 1 then
    	saveImage(output, 'output', iter, epochSaveDir, 'output')
    	for i = 1, 4 do
	    	quicksaveImage(output[2]:select(2, i), 'output[2]', iter, epochSaveDir, 'output', i)
	    end

    	saveImage(target, 'target', iter, epochSaveDir, 'output')
--    	print('size of target[1]:',target[1]:size())
    	for i = 1, 4 do
	    	quicksaveImage(target[2]:select(2, i), 'target[2]', iter, epochSaveDir, 'output', i)
	    end    	
	end


	err = criterion:forward(output, target)
	
	print("\titer",t, "valid score:", err)	
	err = criterion:forward(output, target)

	totalerror = err + totalerror
	
end
    print("\tvalidation total is ", totalerror)
end

--- =============================
--- ==============================

for k = 1, 200 do -- max epoch = 299

	if k == 150 then
		learningRate = 1e-7
	end

	if k == 100 then
		learningRate = 1e-6
	end

	if gpuflag then 
		checkMemory()
		print('move to GPU')
        encoder_0:cuda()
		repeatModel:cuda() 
		checkMemory()
	end
    	if wrapOpticalFlow then   
    		print('test optical flow')
    		-- local optical_flow = mainBranch.modules[1].modules[1].modules[2].modules[3].modules[5].output -- 4 2 50 50
		end
		
	if k == 1 then 
		
    	if wrapOpticalFlow then   
    		print('test optical flow')
    		local optical_flow = flowGridGenerator.modules[3].modules[5].output
		end

		print('test train ')
		checkMemory()
		testFlag = true
		valid()	
	   	saveModel(modelDir, encoder_0, 'test_encoder_0', k)
	   	saveModel(modelDir,repeatModel, 'test_repeatModel', k)	
	   	print("save model pass test")
		train() 

		print('test valid')
		checkMemory()
		testFlag = true


	   	print('test model save, move to CPU')
	   	checkMemory()
	--        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
        encoder_0:float()
		repeatModel:float() 

		checkMemory()
	   	saveModel(modelDir, encoder_0, 'test_encoder_0', k)
	   	saveModel(modelDir,repeatModel, 'test_repeatModel', k)
	   	print('done modelSave')
	   	checkMemory()
		print('move to GPU')
		if gpuflag then
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
		if math.fmod(k , 5) == 1 then
	        saveModel(modelDir,encoder_0, 'encoder_0', k)
		   	saveModel(modelDir, repeatModel, 'repeatModel', k)
	    end
	   	print('modelSave')
		
		checkMemory()
		print('move to GPU')
        encoder_0:cuda()
		repeatModel:cuda() 
	end

end
