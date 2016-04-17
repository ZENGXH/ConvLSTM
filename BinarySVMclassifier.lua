require 'nn'
local BinarySVMclassifier, parent = torch.class('nn.BinarySVMclassifier', 'nn.Module')

function BinarySVMclassifier:__init(dimension)
	parent.__init(self)
   
   outputSize = 1
   inputSize = dimension
   self.weight = torch.Tensor(outputSize, inputSize):uniform(-1,1)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize):fill(0) 

end

function BinarySVMclassifier:updateOutput(input)
	-- target[N, 1]: torch.Tensor(batchSize, 1) = [-1, 1, -1, 1, -1]
	-- input[N, D]: output = input[N, D] x (W[1, D])^T
	-- outputs[N, 1] => [1, -1, 1 ..]
	self.outputs = torch.Tensor(input:size(1)):fill(0) -- [N, 1]
	for batch_i = 1, input:size(1) do
		local input_i = input[batch_i]
		input_i:resize(1, input_i:size(1))
--		print(input_i)
--		print(self.weight:t())
		local score = torch.mm(input_i, self.weight:t())
		score:resize(1)
--		print(score[1])
		if score[1]+self.bias[1] >= 0 then 
			self.outputs[batch_i] = 1
		else
			self.outputs[batch_i] = -1
		end
	end
--	print('update output to be')
	return self.outputs
end

function BinarySVMclassifier:accGradParameters(input, outputs,  target)
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
	self.gradBias = torch.Tensor(1):fill(0)
	self.gradParameters = w.new():resizeAs(w):fill(0) -- [1, D]
	-- local subgradParameters = torch.Tensor(input:size(1)):fill(0)
	local subgradParameters = w.new():resizeAs(w):fill(0)

    for batch_i = 1, input:size(1) do
    	local subgradParameters_i = 0
--	print('grad parameters _i')
--	print(self.outputs)
--	print(target)

    	if self.outputs[batch_i] * target[batch_i] < thre then
			subgradParameters_i = torch.mul(input[batch_i], - self.outputs[batch_i]) -- input[batch_i] 
			-- outputs[batch_i] should be number
			-- return [1, D] tensor
--			print(self.gradBias)
			self.gradBias[1] = self.gradBias[1] + 1
	end
		subgradParameters = subgradParameters + subgradParameters_i
    end
--	print(1/input:size(1))
--	print(w)
--	print(subgradParameters)

	self.gradParameters = torch.mul(w, lambda) + torch.mul(subgradParameters,1/input:size(1) )
	-- shape [1, D]
	self.gradParameters = self.gradParameters
	self.gradWeight = self.gradParameters
	-- nt(self.gradWeight)
	-- return self.gradWeight
end


function BinarySVMclassifier:updateGradInput(input, gradOutput, target)
    -- GradInput = -W/N if y_i * input_i * W^T > 1, else 0
	local w = self.weight
    self.gradInput =  input.new():resizeAs(input):fill(0) -- [N, D]
    local thre = 1
    local resize_flag = 0
    local target = gradOutput
--	print(target)

    if target:size():size() > 1 then
    	resize_flag = 1
    	target:resize(target:size(1))
    end

    for batch_i = 1, input:size(1) do
    	local gradInput_i = w.new():resizeAs(w):fill(0)
        -- for support vector: 
    	if self.outputs[batch_i] * target[batch_i] < thre then
		print('support v')
--		print(target[batch_i])
--		print(w)
    		-- gradInput_i = - target[batch_i] * w
    		gradInput_i = torch.mul(w,- target[batch_i])
		local s = torch.Tensor{1 - self.outputs[batch_i] * target[batch_i]}
--		print(s)
--                print(f)
		f = torch.Tensor{f[1] +  s[1]}
                print('f update')
		print(f)
		-- f = f + torch.Tensor{1 - self.outputs[batch_i] * target[batch_i]}
    	end
    	self.gradInput[batch_i] = gradInput_i
    end
end

function BinarySVMclassifier:backward(input, gradOutput, scale)
   f = torch.mm(self.weight, self.weight:t())
   f:resize(1)   
--   print('init f and gradOutput')
--   print(f)
--   print(gradOutput)
   scale = scale or 1
   target = gradOutput
   self:updateGradInput(input, gradOutput, target)
   self:accGradParameters(input, outputs, target)
   return self.gradInput
end

function BinarySVMclassifier:backwardUpdate(input, gradOutput, lr)
   lr = 1
   self:updateGradInput(input, gradOutput, target)
   self:accUpdateGradParameters(input, outputs, target)
   return self.gradInput
end
