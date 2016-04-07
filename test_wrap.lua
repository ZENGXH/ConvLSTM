-- warp with optical flow

unpack = unpack or table.unpack
onMac = true
saveOutput = true
resumeFlag = false
resumeIter = 1
verbose = true
display = true
require 'paths'
local modelDir = 'image/add_clamp/'
local imageDir = 'image/simpleWrap/add_clamp/'

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


local saveInterval = 20
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


function saveImage(figure_in, name, iter, epochSaveDir, type)
	local figure = torch.Tensor()

	if not torch.isTensor(figure_in)  then 
		-- print('size of figure table', name, figure_in)
		if table.getn(figure_in) == 1 then
			figure = figure_in[1]:clone()
		end
    else
    	figure = figure_in:clone()
	end

	if torch.isTensor(figure) and type == 'flow' then
        -- print('size of figure', name, figure:size())
		local img = figure:clone():float() -- :resize(1, 100, 100)
		-- img[1] = img[1]/img[1]:max()
		-- img[2] = img[2]/img[2]:max()
		-- img[3] = img[3]/img[3]:max()

   	        if type == 'flow' and display then
		    print('min and max of '..name..' is', img:min(),img:max())
		    	print(img[1]:mean(),img[2]:mean(),img[3]:mean())
		    -- print('size :'img:size())
   	        	image.display(img)
   	        end		
      
		
	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-.png',  img/img:max())
	elseif 
			type == 'output' and torch.isTensor(figure) then
    	-- for numsOfOut = 1, table.getn(figure) do
	    	local img = torch.Tensor()
    		if figure:size(2) == 4 then 

    			--for k = 1,4 do
    			--	local sub = figure:select(2,k)
	    		--	sub = sub/sub:max()
	    		--	figure[1][k] = sub
				--end

   	            img = dataMask:quick4Mask3Dbackward(figure):float() -- :resize(1,100,100)
   	        else 
   	        	img = figure:clone()
            end
   	        -- print('',name, img)
   	        -- img:squeeze()


   	        if name == 'output' and display then
		        print('min and max of '..name..' is', img:min(),img:max(), img:mean())
		    -- print('size :'img:size())
   	        	image.display(img)
   	        end
    	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
    	-- end 

	elseif type == 'gard' then
		for numsOfOut = 1, table.getn(figure) do
   	    	local img = dataMask:quick4Mask3Dbackward(figure):float():resize(1, opt.nFiltersMemory[2] * 100, 100)
   	    	img = img:mul(1/img:max())
    	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img/img:max())
    	end  
	end
end

function quicksaveImage(figure_in, name, iter, epochSaveDir, type, numsOfOut)
	if type == 'output' then
    	local img = figure_in:clone():float()
    	-- img = img:mul(1/img:max())
	    image.save(epochSaveDir..'quick-iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img)
	elseif type == 'grad' then
    	local img = figure_in:clone():float()
    	-- img = img:mul(1/img:max())
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



if gpuflag then 
	print('gpu flag on') 
else 
	print('gpu flag off')
end

local iter = 0

if resumeFlag then

	print('==> load model, resume traning')
  	encoder_0 = torch.load(modelDir..'test_encoder_0iter-'..resumeIter..'.bin')
  	convForward_4 = torch.load(modelDir..'test_convForward_4iter-'..resumeIter..'.bin')
  	print('load model done')
else
	print('==> build model for traning')

	encoder_0 = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, true, 3)):float()-- batchsize

	local lstm_params0, lstm_grads0 = encoder_0:getParameters()
	lstm_params0:uniform(-0.08, 0.08)

    local reshape = nn.View(opt.batchSize, opt.fakeDepth, opt.memorySizeH, opt.memorySizeW )

	local branch_up = nn.Sequential() -- do convolution to 1*17*50*50 input , output size == itarget
	                               :add(nn.View(opt.batchSize, opt.nFiltersMemory[2], opt.width, opt.width ))
	                               :add(nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
							       :add(nn.View(opt.batchSize, opt.fakeDepth, opt.memorySizeH, opt.memorySizeW )) -- 1, 4, 50, 50 => 1 50 50 4
							      
							       :add(nn.Transpose({2,4}, {2,3})):float() -- batchSie, H, W, fakeDepth
							       -- output b1 h50 w50 d4: 
							       :add(nn.Transpose({1,4})) -- make fake Depth batchSIZE
							       -- tobe tempBatch4 h50 w50 d1:

	-- local interface = nn.Identity() -- nn.SelectTable(opt.input_nSeq)


	--branch_up:add(nn.JoinTable(1)) -- along width direction
	--memory_branch:add(nn.JoinTable(1)) -- along width direction
	--[[if not wrapOpticalFlow then

		print('not optical flow warpping')
		local concat = nn.ConcatTable()
		concat:add(branch_up):add(branch_up)
		local wrapConcat = nn.Sequential():add(concat):add(nn.SelectTable(1)):add(nn.Transpose({1,3}, {2,3}))
		-- local sampler = nn.Sequencer(wrapConcat):float()
		convForward_4 = nn.Sequential():add(interface):add(wrapConcat):float()
	--	convForward_4 = nn.Sequential():add(nn.FlattenTable()):add(sampler):float()
		convForward_4 = convForward_4:float()
	else
		
		local memory_branch = nn.Sequential()
										:add(nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
	                                    :add(reshape)
	                                    :add(flow):float() 		
		local memory_branch_sub = nn.Sequential()
										:add()
	                                    -- :add(nn.View(opt.batchSize, 1, opt.memorySizeH, opt.memorySizeW ))
	                                    :add(flow):float() ]]
	    local pre_memo = nn.Sequential()
									:add(nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1))
	    							--:add(nn.SplitTable(2))
	    							--:add(nn.SelectTable(1))
	    							--:add(memory_branch_sub)
	    							:add(nn.Transpose({1,2})) 
		    							-- make fakeDepth batch 
		    							-- B1 D4 H50 W50 => B4 D1 H50 W50
	    							-- :add(nn.JoinTable(1))
		-- for input of flow: B4 D1 H50 W50, depth of first conv should be opt.batchSize -> others
	    							:add(flow) 
	    							-- :add(nn.Sequencer(memory_branch_sub))  
	    							-- :add(nn.JoinTable(1))

		local concat = nn.ConcatTable()
		--concat:add(branch_up):add(memory_branch)
		concat:add(branch_up):add(pre_memo)
		predict_branch = branch_up

		flow_warp = nn.ParallelTable():add(pre_memo):add(nn.Transpose({1,2},{2,3},{3,4}))

		flow_warp:add(nn.BilinearSamplerBHWD()) -- output b h w d: 4 50/h 50/w 1
				 :add(nn.Transpose({3,4}, {2,3})) 
									-- bdhw 4(bathSize which is fake depth) 1 50/h 50/w
				 :add(nn.Transpose({1, 2})) -- return to 4 as depth
--		combine_branch = nn.Sequential():add()
		-- expected output: b h w d
		print('branch_up')
		print(branch_up)
		local wrapConcat = nn.Sequential()
									:add(concat)
									:add(nn.BilinearSamplerBHWD()) -- output b h w d: 4 50/h 50/w 1
									:add(nn.Transpose({3,4}, {2,3})) 
									-- bdhw 4(bathSize which is fake depth) 1 50/h 50/w
									:add(nn.Transpose({1, 2})) -- return to 4 as depth
		-- local sampler = nn.Sequencer(wrapConcat):float()
	    -- add sampler
		convForward_4 = nn.Sequential()
								--:add(interface)
								:add(wrapConcat):add(nn.Clamp(0.00,1.00)):float()
		-- convForward_4 = convForward_4:float()
	--[[end]]
	print(convForward_4)
	br = convForward_4.modules[1].modules[1].modules[1]
	memo = convForward_4.modules[1].modules[1].modules[2]
	fn = memo.modules[3]

	sampler = convForward_4.modules[1].modules[2]
	local lstm_params4, lstm_grads4 = convForward_4:getParameters()
	lstm_params4:normal(paraInit, std)


	--convForward_4:add(nn.Transpose({1,3}, {2,3}))
	--             :add(nn.View(opt.imageDepth, opt.imageH, opt.imageW, opt.output_nSeq))
	----------------------------------------------------------------------------
end


print('encoder_0')
print(encoder_0)

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

  convForward_4:remember('both') 
  convForward_4:training()
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

    convForward_4:zeroGradParameters()
    convForward_4:forget()

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
    
    inputTable4 = output0[opt.input_nSeq]
    -- print('output of encoder_0: ', output0)

	output = convForward_4:forward(inputTable4)

    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	if wrapOpticalFlow then  
    		local optical_flow = convForward_4.modules[1].modules[1].modules[2].modules[3].modules[5].output -- 4 2 50 50
    		print('optical flow is 4 * nDimension', optical_flow[1]:nDimension())

    		for i = 1, 4 do
    			local flow_depth = optical_flow[i]
 		   	    local imflow = flow2colour(flow_depth)
            	saveImage(imflow, 'flow-n'..i, iter, epochSaveDir, 'flow')
	        end
        end
        
       -- saveImage(output0, 'output0', iter, epochSaveDir, 'output')
    end

	--criterion = nn.SequencerCriterion(nn.MSECriterion()):float()
	criterion = nn.MSECriterion():float()
	if gpuflag then criterion:cuda() end
     
    local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.fakeDepth, opt.inputSizeH, opt.inputSizeW):float()
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
	if opt.output_nSeq == 1 then
		target = target[1]
	end
--    if gpuflag then targetSeq = targetSeq:cuda() end
    if saveOutput and math.fmod(t , saveInterval) == 1 or t == 1 then
    	saveImage(output, 'output', iter, epochSaveDir, 'output')
    	for i = 1, 4 do
	    	quicksaveImage(output:select(2, i), 'output', iter, epochSaveDir, 'output', i)
	    end

    	saveImage(target, 'target', iter, epochSaveDir, 'output')
    	print('size of target:',target:size())
    	for i = 1, 4 do
	    	quicksaveImage(target:select(2, i), 'target', iter, epochSaveDir, 'output', i)
	    end    	
	end




	print('criterion')
	assert(target:size(1) == output:size(1))
	assert(target:size(2) == output:size(2))
	assert(target:size(3) == output:size(3))
	assert(target:size(4) == output:size(4))
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
    	for i = 1, 4 do
	    	quicksaveImage(gradOutput:select(2, i), 'gradOutput', iter, epochSaveDir, 'grad', i)
	    end
        saveImage(gradOutput, 'gradOutput', iter, epochSaveDir, 'grad')
	end
	gradOutput = {}


	gardOutput_encoder_0 = convForward_4.gradInput

	encoder_0:backwardUpdate({inputTable[opt.input_nSeq]}, {gardOutput_encoder_0}, learningRate)
	encoder_0.module:forget()
	encoder_0:zeroGradParameters()


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


    encoder_0:zeroGradParameters()
    encoder_0:forget()

    convForward_4:zeroGradParameters()
    convForward_4:forget()

	output0 = encoder_0:updateOutput(inputTable)

    inputTable4 = output0[opt.input_nSeq]

	output = convForward_4:forward(inputTable4)

    if saveOutput and math.fmod(t , saveInterval) == 1 and t > 1 then
    	if wrapOpticalFlow then  
            local optical_flow = convForward_4.modules[2].modules[1].modules[2].modules[3].modules[7].output
    	    local imflow = flow2colour(optical_flow)
            saveImage(imflow, 'flow', iter, epochSaveDir)
        end
        -- saveImage(output, 'output', iter, epochSaveDir, 'output')
        saveImage(output0, 'output0', iter, epochSaveDir, 'output')
    end

	--criterion = nn.SequencerCriterion(nn.MSECriterion()):float()
	criterion = nn.MSECriterion():float()
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

	target = target[1]
	err = criterion:forward(output, target)


	err = criterion:forward(output, target)
    print('valid score:',err)
	totalerror = err + totalerror
	
end
    print("\tvalidation total is ", totalerror)
end

--- =============================
--- ==============================

for k = 1, 2000 do -- max epoch = 299

	if k == 500 then
		learningRate = 1e-5
	end

	if k == 1000 then
		learningRate = 1e-7
	end

	if gpuflag then 
		checkMemory()
		print('move to GPU')
        encoder_0:cuda()
		convForward_4:cuda() 
		checkMemory()
	end
    	if wrapOpticalFlow then   
    		print('test optical flow')
    		local optical_flow = convForward_4.modules[1].modules[1].modules[2].modules[3].modules[5].output -- 4 2 50 50
		end
		
	if k == 0 then 
		
    	if wrapOpticalFlow then   
    		print('test optical flow')
    		local optical_flow = convForward_4.modules[1].modules[1].modules[2].modules[3].modules[5].output -- 4 2 50 50
		end

		print('test train ')
		checkMemory()
		testFlag = true
		train() 

		print('test valid')
		checkMemory()
		testFlag = true
		-- valid()

	   	print('test model save, move to CPU')
	   	checkMemory()
	--        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
        encoder_0:float()
		convForward_4:float() 

		checkMemory()
	   	saveModel(encoder_0, 'test_encoder_0', k)
	   	saveModel(convForward_4, 'test_convForward_4', k)
	   	print('done modelSave')
	   	checkMemory()
		print('move to GPU')
		if gpuflag then
	        encoder_0:cuda()
			convForward_4:cuda() 
		end
		print('run all valid first')
		-- valid() -- run all valid first
		print(c.red 'pass test!')
	end

		train()
		-- valid()

	if gpuflag then 
		encoder_0:float() 
		convForward_4:float() 
		if math.fmod(k , 2) == 1 then
	        saveModel(encoder_0, 'encoder_0', k)
		   	saveModel(convForward_4, 'convForward_4', k)
	   end
	   	print('modelSave')
		
		checkMemory()
		print('move to GPU')
        encoder_0:cuda()
		convForward_4:cuda() 
	end

end
