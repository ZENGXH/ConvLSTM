require 'logging.file'
require 'nn'
require 'torch'
local flag = 0
local PartialConnected, parent = torch.class('nn.PartialConnected', 'nn.Module')
-- dbg = require("debugger")
local logger = logging.file("test_partialconnect.log")
local testp1 = -1 -- vali on forward
local testp2 = 200 -- vali on backward
local para_rocord = 0
  
function PartialConnected: __init( batchSize, inputSize, outputSize)
        parent.__init(self)

        self.batchSize = batchSize
        self.inputSize = inputSize

        self.outputSize = outputSize
        self.num_model = inputSize/outputSize
        num_model = self.num_model

        self.input_pmodel = inputSize/num_model --dimension of feature
        input_pmodel = self.input_pmodel
        self.output_pmodel = 1
        output_pmodel = self.output_pmodel
        print('batchSize, num of model, input_pmodel, output_pmodel:')
        print(self.batchSize, self.num_model, self.input_pmodel, self.output_pmodel)
        self.model = {}
        input_split_flag = false
        gradOutput_split_flag = false

        for i=1, num_model do
                table.insert(self.model, self:createModel(self))
        end
--      print(model)
        -- #
        -- three in a mask
        self.weight = torch.Tensor(1, num_model * self.input_pmodel):uniform(-1,1)
        self.bias = torch.Tensor(num_model):uniform(-1,1)
        self:updateModel2self()
        self:getParaFromNet()
        self:reset()

end

function PartialConnected:getParaFromNet()
        for i = 1,num_model do
                self.model[i].weight = self.weight[{{},{(i-1)*input_pmodel + 1, i*input_pmodel}}]
                self.model[i].bias = self.weight[{{},{i}}]
        end
end

function PartialConnected:createModel()
        -- inputSize and be {N, 2}
        local submodel = {}
        submodel.input = torch.Tensor(self.batchSize, self.input_pmodel)

        submodel.weight = torch.Tensor(self.input_pmodel,self.output_pmodel)
        submodel.gradWeight = torch.Tensor(self.input_pmodel, self.output_pmodel)

        submodel.bias = torch.Tensor(1) -- each model have one bias, intotal num_model
        submodel.gradBias = torch.Tensor(1) --
        submodel.gradInput  = torch.Tensor(self.batchSize, self.input_pmodel)
        submodel.output = torch.Tensor(self.batchSize, self.output_pmodel)
        submodel.gradOutput = torch.Tensor(self.batchSize, self.output_pmodel)

        return submodel
end

function PartialConnected:updateModel2self()
--      self.weight = self.model[1].weight
--      self.bias = self.model[1].bias
        self.gradWeight = self.model[1].gradWeight -- [outputsize, inputsize]
	self.gradBias = self.model[1].gradBias --[num_model, 1]
        self.gradInput = self.model[1].gradInput --[N, D]
	for i = 2, num_model do
--		self.weight = torch.cat(self.weight, self.model[i].weight)
--		self.bias = torch.cat(self.bias, self.model[i].bias)
		self.gradWeight = torch.cat(self.gradWeight, self.model[i].gradWeight)
		self.gradInput = torch.cat(self.gradInput, self.model[i].gradInput)
		
		self.gradBias = torch.cat(self.gradBias, self.model[i].gradBias)
	end	
--	self.weight:resize(8)
--	print(batchSize)
--	inputSize = 100
--	self.weight:resize(1, num_model * input_pmodel)
	self.gradWeight:resizeAs(self.weight)
	self.gradBias:resize(num_model)
        para_record = self.weight	
--	print('self.weight', self.weight)
--	if flag > testp2 then
--	print('self parameters  weight, bias,')
--	print(self.gradWeight[{{1,2}}])
--	print(self.weight[{{1,2}}])
--	print(self.gradBias[{{1}}])
--	print(self.bias)	
--	print('input and output')
--	print(self.model[1].input)
--	print(self.model[2].input)
--	print(self.model[1].output)
--	print(self.model[2].output)
--	end
--	flag = flag + 1
end


function PartialConnected:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else

      stdv = 1./math.sqrt(self.model[1].weight:size()[1])
   end

   for i = 1, num_model do
      self.model[i].weight:uniform(-stdv, stdv)
      self.model[i].bias:uniform(-stdv, stdv)
   end

   return self
end

function PartialConnected:backward(input, gradOutput, scale)
   scale = scale or 1
--	print('in partial backward start gradOutput is ',gradOutput)
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
--	print('in partial backward done, gradInput is', self.gradInput)
   self.gradInput:resize(self.gradInput:size(1), self.gradInput:size(2), 1, 1)
   return self.gradInput
end

function PartialConnected:forward(input)
   local input_resize = torch.Tensor(input:size(1), input:size(2)):copy(input)
   
   return self:updateOutput(input_resize)

end

function PartialConnected:updateOutput(input) 
	-- input size； BATCH * D
	-- input dimension should be 12 x 1 vector
	-- output = torch.Tensor(4):fill(0) -- initilaize
	self:getParaFromNet()
	if(not input_split_flag) then self:splitInput2Sub(input) end
	if(para_record ~= self.weight) then print('weight change!') end
	if(input:dim() ~= 1) then 
		-- reshape()
		for i = 1, #self.model do -- 1 to 4
			self.model[i].input:resize(self.batchSize, input_pmodel)
			self.model[i].weight:resize(1, input_pmodel)
--			print('#model:', i)
			local weight_i = torch.Tensor(1, input_pmodel):copy(self.model[i].weight):float()
--			print('input: ', self.model[i].input)
--			p(self.model[i].bias)
--			p(self.model[i].output)
			-- self.model[]
--			print(weight_i)
			self.addbuffer = torch.Tensor(self.batchSize, 1):fill(1):float() -- scala in this case
			-- self.model[i].output:addmm(0, 1, self.model[i].weight, self.model[i].input)
			-- self.model[i].output[1] =torch.dot(self.model[i].weight, self.model[i].input)
			
			local output_resize =torch.mm(self.model[i].input, weight_i:t())
			self.model[i].output = output_resize:resize(self.batchSize, 1)
			if i< testp1 then
				print(i)
				print('weight',self.model[i].weight)
				print('input',self.model[i].input)
--			print(' output weight times input')
--			print(self.model[i].output)
			end
--			print(self.model[i].weight)
--			print(type(self.model[i].output))
			--self.model[i].output:addr(1, self.model[i].output,1, self.model[i].bias, self.addbuffer)
--			print(self.model[i].bias)
			-- self.model[i].output = torch.addmm(1, self.model[i].output, self.model[i].bias[1], self.addbuffer) 
			
			local bias_i = self.model[i].bias[1][1]
--			print(bias_i)
			self.model[i].output = torch.add(self.model[i].output, bias_i, self.addbuffer)
			self.model[i].output:resize(self.batchSize, self.output_pmodel)
			if i< testp1 then
			print('add')
				print('bias',self.model[i].bias)
				print('output',self.model[i].output)
			end
			-- use torch.addmv([res,] [beta,] [v1,] vec1, [v2,] mat, vec2)
			-- torch.addmv(1Db_vec(N,D), X(N,D), W(D))
			-- res = (beta * res) + (v1 * vec1) + (v2 * (mat * vec2))
			-- beta: momentumn, v1: bias, v2 = 1, mat: X, vec2: weight
--			print(self.model[i].output)
		end
	
	else
		print('dimension error')
		-- sli = input:reshape(input:size(1),4,2) 
		-- for i = 1, #sli[1] do -- 1 to 4
		-- output[i] = bias[i] + weight[i] * sli[i] -- use torch.addmv
		-- output:addmv()
	end
	-- cat output
	local out = self.model[1].output

	for i = 2, #self.model do
		
		out = torch.cat(out, self.model[i].output)
	end
	
	self.output = out:resize(self.batchSize, self.outputSize, 1, 1)
--	print('iterartion done, input and output')
--	print(input)
--	print(out)
--	logger:info('input and output is')
--	logger:info(input)
--	logger:info(out)
--	print('=======================')
	return self.output -- 4x1
end
--[[
function PatialConnected:backward(intput, gradOutput)
	bp, gradOutput
		(pass by the back layer 
		base to the output of current layer and target) 
		return by 
			`MSECriterion_updateOutput`
				use new output 
				run computeCost and gradient again then call
			`MSECriterion_updateGradInput` 
				=> store the result of last computation of 
				the gradients of the loss function(by criterion.backward)
	   
	   function task: call 
			`updateGradInput(input, gradOutput)` 
			`accGradParameters(input,gradOutput,scale)`
				if only one layer, gradOutput is exactly: 
					gradOutput is gradients compute from criterion
					= \partial{(y - W^T X)} / \partial{W} 
					= 1/N X'* (y - W^T X)
					= 1/N X'* (target - outputResult), 
					X is the input of the final layer
				if more then one layer, by BP,
					gradient wrt to parameters
					= gradOutput 
	 
end --]]

function PartialConnected:accGradParameters(input, gradOutput, scale)
	--[[ 
		`GradParameter` means grad wrt to weight and grad wrt bias
		`scale` is a scale factor 
			that is multiplied with the gradParameters before being accumulated.
			(not momentumn! here is calculating current gradient, not the g for parameter updating)
		`gradient wrt to W` = gradWeight 
			= GET * \patial{output}/\partial{para} 
			= GET * X 
			= input * gradOutput	 
		
		gradient wrt to bias:
			= GET * \partial{output = W^T + b}/\partial{b}
			= GET
		bias = step * GET
	--]]

	self:getParaFromNet()
	scale = scale or 1
	if(not input_split_flag) then self:splitInput2Sub(input) end
	if(not gradOutput_split_flag) then self:splitGradOutput2Sub(gradOutput) end

	if(input:dim() ~= 1) then 

		--[[
			[res] torch.addr([res,] [v1,] mat, [v2,] vec1, vec2)
				= mat(M,N) + vec1(M,1)vec1'(1,N)
			notice that, if write as: 
				a(D,1):addr(scale, gradOutput(D, 1), input(D, 1))
				=> a = scale * a + gradOutput(D, 1) input'(1, D)
			ie, gradWeight(12 = D1(3) + D1(3) + D3(3) + D4(3), 1) 
				= (gradWeight * scale -- ) + gradOutput(D, 1) x input
		--]]

		-- check dimention: reshape() first dim is #model
		-- sli = input:reshape(4, 2) -- (#model, #ft)
		-- self.gradWeight:reshape(4,3) -- 
		-- gradOutput: reshape(4, 1)

		for i = 1, #self.model do
			-- ?? self.weight[i].addr(scale, gradOutput[i], sli[i])
		        self.model[i].gradOutput:float()	
			self.model[i].gradWeight = torch.mm(self.model[i].input:t(), self.model[i].gradOutput) 
			-- self.model[i].gradBias:addmv(scale, self.model[i].gradOutput:t(), self.model[i].addBuffer) 
			self.model[i].gradBias = self.model[i].gradOutput 
		end	-- in the case of output is scala
	else 
		print('input dimension should not be one')
	end

	self:updateModel2self()

end

function PartialConnected:updateGradInput(input, gradOutput)
	--[[ 
		compute gradient of the module wrt its own input
		return: gradInput, 4 model, each in dimension 2 + 1, 
		4 X 3 dimension 
		gradOutput in shape(#current layer output, 1)
		
		pass to the previous layer: 
		PASS 
			= GET * \partial{output}/\partial{input}
			= GET * W
	--]]
	self:getParaFromNet()
	if(not input_split_flag) then self:splitInput2Sub(input) end
	if(not gradOutput_split_flag) then self:splitGradOutput2Sub(gradOutput) end

	if input:dim() ~= 1 then
		for i = 1,#self.model do
			-- self.model[i].gradInput:addmv(0, 1, self.model[i].gradOutput, self.model[i].weight)
--			print(self.model[i].gradOutput)
--			print(self.model[i].weight)
			-- self.model[i].gradInput = torch.Tensor(2,1):fill(0):float	
			self.model[i].weight:resize(input_pmodel,output_pmodel)

			self.model[i].gradInput = torch.mm(self.model[i].gradOutput, self.model[i].weight:t())
--			self.model[i].weight:resize(input_pmodel,)

		end
	end

	self:updateModel2self()
end


function PartialConnected: splitInput2Sub(input)
--	input_split_flag = true
	for i=1, #self.model do
--		print(self.model[i].input:size())
--		print(i)
--		self.model[i].input:resize(batchSize, input_pmodel,1,1)
		self.model[i].input = input[{{}, 
			{1 + (i-1) * input_pmodel, i * input_pmodel}}]:float()

--		print(self.model[i].input:size())
		self.model[i].input:resize(self.batchSize, input_pmodel)
	end
end

function PartialConnected:splitGradOutput2Sub(gradOutput)
--	gradOutput_split_flag = true
	for i=1, #self.model do
		self.model[i].gradOutput = gradOutput[
			{{1 + (i-1) * output_pmodel, i * output_pmodel}}]
		self.model[i].gradOutput:resize(self.batchSize, output_pmodel)
	end
end

function p(x)
	print(x:size())
end
