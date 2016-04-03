require 'nn'
local mulBinarySVMclassifier, parent = torch.class('nn.mulBinarySVMclassifier', 'nn.Module')

function mulBinarySVMclassifier:__init(dimension)
	parent.__init(self)
   
   outputSize = 4
   numClass = 4
   inputSize = dimension
   self.weight = torch.Tensor(outputSize, inputSize):uniform(-1,1)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize):fill(0) 

end

function mulBinarySVMclassifier:updateOutput(input)

	-- target[N, 4]: torch.Tensor(batchSize, 4) = [{ {1, -1, -1, -1}(class 1), {-1, 1, -1, -1}(class2) ..}]
	-- input[N, D]: output = input[N, D] x (W[1, D])^T
	-- outputs[N, 4] => [ {1, -1,..}, {1 ..}, ] need to convert the output for confusion matrix

-- or

	-- target[N, 1]: torch.Tensor(batchSize, 1) = [2,3,1,4,2,...]
	-- intput is the same as above
	-- outputs[N, 1]: torch.Tensor(batchSize, 1) = [2,3,4,5,1,2,] based on the 4 score

	self.outputs = torch.Tensor(input:size(1)):fill(-1):float() -- [N, 1] for method 2
	self.binary_outputs = torch.Tensor(input:size(1), outputSize):fill(-1):float()
--	self.binary_score = torch.Tensor(intput:size(1), outputSize):fill(0)
	for batch_i = 1, input:size(1) do
		local input_i = input[batch_i]
		input_i:resize(1, input_i:size(1))
--		print(input_i)
--		print(self.weight:t())
		local score = torch.mm(input_i, self.weight:t()) --[four score]
		score:resize(outputSize)
--		print(score[1])
--		binary_output = torch.Tensor(input:size(1)):fill(-1)

		local binary_score = - math.huge -- init to be minus inf

		for j=1,4 do
		  if score[j]+self.bias[j] >= 0 then 
		    self.binary_outputs[{{batch_i},{j}}] = 1		      
		    if score[j]+self.bias[j] > binary_score then
	              self.outputs[batch_i] = j -- take higher score as output class
		      binary_score = score[j]+self.bias[j]
		    end
		  end
		end

		if self.outputs[batch_i] == -1 then
		  for j = 1,4 do
			if score[j] + self.bias[j] > binary_score then 	
	                  self.outputs[batch_i] = j -- take higher score as output class
		          binary_score = score[j]+self.bias[j]
			end
		end
		end
		
	end
	--print('update output to be')
	--print(self.outputs)
	return self.outputs
end

function mulBinarySVMclassifier:accGradParameters(input, outputs,  target)
	-- w: [1, D]
	-- input[N, D]: output = input[N, D] x (W[1, D])^T
	-- penalizedSVMcost = 1/N sum_i max(0, 1 - y_i * input_i * w^T) + lambda * w^2
-- = 1/N * [max(0, 1 - y_1 * input_1 * W^T) + max(0, 1 - y_1 * input_i * W^T) + ...] 
	-- 	 + lambda * W^T
	-- GradParameters = lambda * w + 1/N [(0 or - y_1 * input_1) + (0 or - y_2 * input_2) + ...]
    local resize_flag = 0
    if target:size():size() > 1 then
        resize_flag = 1
        target:resize(target:size(1))
    end

	local w = self.weight
	local thre = 1
	local lambda = 0.01
	self.gradBias = torch.Tensor(numClass):fill(0)
	self.gradParameters = w.new():resizeAs(w):fill(0) -- [1, D]
	-- local subgradParameters = torch.Tensor(input:size(1)):fill(0)
	local subgradParameters = w.new():resizeAs(w):fill(0)

    local binary_target = torch.Tensor(input:size(1), numClass):fill(-1):float()
    
    for batch_i = 1, input:size(1) do
        binary_target[{{batch_i},{target[batch_i]}}] = 1
    	local subgradParameters_i = w.new():resizeAs(w):fill(0) -- parameters: [numofclass:4 , dimension_input]
--	print('grad parameters _i')
--	print(self.outputs)
--	print(target)
        for j = 1,4 do

    	  local mar = torch.mm(self.binary_outputs[{{batch_i},{j}}], binary_target[{{batch_i},{j}}])
	  mar:resize(1)
	  if mar[1] < thre then
--	  print(self.binary_outputs[{{batch_i},{j}}])
--	print(input[batch_i])
	     subgradParameters_i[{{j},{}}] = torch.mm(self.binary_outputs[{{batch_i},{j}}],input[{{batch_i},{}}]):resizeAs(subgradParameters_i[{{j},{}}]) 
	     

			-- input[batch_i] 
			-- outputs[batch_i] should be number
			-- return [1, D] tensor
--			print(self.gradBias)
--		print(self.gradBias)
	     self.gradBias[j] = self.gradBias[j] + 1
	  end
	end
		subgradParameters = subgradParameters + subgradParameters_i*(-1) -- -y*x
    end
--	print(1/input:size(1))
--	print(w)
--	print(subgradParameters)

--	self.gradParameters = torch.mv(w, torch.Tensor{lambda}) + torch.mv(subgradParameters, torch.Tensor{1/input:size(1)})
	self.gradParameters = w*(lambda) + subgradParameters*(1/input:size(1))
	-- shape [1, D]
	self.gradParameters = self.gradParameters
	self.gradWeight = self.gradParameters
	-- nt(self.gradWeight)
	-- return self.gradWeight
end


function mulBinarySVMclassifier:updateGradInput(input, gradOutput, target)
    -- GradInput = -W/N if y_i * input_i * W^T > 1, else 0
	local w = self.weight
    self.gradInput =  input.new():resizeAs(input):fill(0) -- [N, D]
    local thre = 1
    local resize_flag = 0
    local target = gradOutput
--	print(target)

    local binary_target = torch.Tensor(input:size(1), numClass):fill(-1):float()
    
    if target:size():size() > 1 then
    	resize_flag = 1
    	target:resize(target:size(1))
    end

    for batch_i = 1, input:size(1) do

        binary_target[{{batch_i},{target[batch_i]}}] = 1

    	local gradInput_i = self.gradInput.new():resizeAs(self.gradInput):fill(0)
        -- for support vector: 
        for j = 1,4 do


    	  local mar = torch.mm(self.binary_outputs[{{batch_i},{j}}], binary_target[{{batch_i},{j}}])
	mar:resize(1)
--    	  if self.binary_outputs[{{batch_i},{j}}] * binary_target[{{batch_i},{j}}][1][1] < thre then
	  if mar[1] < thre then
--		print('support v')
--		print(target[batch_i])
--		print(w)
    		-- gradInput_i = - target[batch_i] * w
    		
--		print(gradInput_i)
	        gradInput_i[{{},{j}}] = torch.mm(binary_target[{{batch_i},{j}}], w[{{j},{}}]):resizeAs(gradInput_i[{{},{j}}])
--    		local s = torch.Tensor{1 - self.outputs[batch_i] * target[batch_i]}
--		print(s)
--                print(f)
--		f = torch.Tensor{f[1] +  s[1]}
--              print('f update')
--		print(f)
		-- f = f + torch.Tensor{1 - self.outputs[batch_i] * target[batch_i]}
    	  end
        end
    	 self.gradInput = self.gradInput +  gradInput_i * (-1)
    end

--    	self.gradInput = gradInput_i * (-1)
end

function mulBinarySVMclassifier:backward(input, gradOutput, scale)
--   f = torch.mm(self.weight, self.weight:t())
--   f:resize(1)   
--   print('init f and gradOutput')
--   print(f)
--   print(gradOutput)
   scale = scale or 1
   target = gradOutput
   self:updateGradInput(input, gradOutput, target)
   self:accGradParameters(input, outputs, target)
   return self.gradInput
end

function mulBinarySVMclassifier:backwardUpdate(input, gradOutput, lr)
   lr = 0.001
   self:updateGradInput(input, gradOutput, target)
   self:accUpdateGradParameters(input, outputs, target)
   return self.gradInput
end
