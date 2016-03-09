--[[
  Convolutional LSTM for short term visual cell
  inputSize - number of input feature planes
  outputSize - number of output feature planes
  rho - recurrent sequence length
  kc  - convolutional filter size to convolve input
  km  - convolutional filter size to convolve cell; usually km > kc  
--]]
local _ = require 'moses'
require 'nn'
require 'dpnn'
require 'rnn'
-- require 'extracunn'

local ConvLSTM, parent = torch.class('nn.ConvLSTM', 'nn.LSTM')

--[[to be copy(64, 64, -- inputSize and outputSize
              5,      -- length of seq
              3, 3,   -- size of kernel for cell and memory
              1, 8,   -- stride and batchsize
              true, 3,-- with cell to gate, kernel size 3
              false)  -- no input for LSTM
]]--
function ConvLSTM:__init(inputSize, outputSize, rho, kc, km, stride, batchSize, cell2gate, ka, inputFlag)
   print("intputsize", inputSize)
   self.kc = kc or 3
   self.km = km or 3
   self.padc = torch.floor(kc/2) 
   self.padm = torch.floor(km/2)
   self.stride = stride or 1
   self.batchSize = batchSize or nil
   self.cell2gate = cell2gate or true
   self.ka = ka or 3
   -- for decoder1, ie layer 2: no input 
   self.inputFlag = inputFlag or true

   parent.__init(self, inputSize, outputSize, rho or 5)
end

-------------------------- factory methods -----------------------------
function ConvLSTM:buildGate()
   -- Note : Input is : {input(t), output(t-1), cell(t-1)}
   
   local gate = nn.Sequential()
   gate:add(nn.NarrowTable(1,2)) -- we don't need cell here

   if(self.inputFlag) then
        local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
    else
        print(" setup LSTM with no input to inputs !! ")
    end 

   local output2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   if(self.cell2gate) then
       local cell2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.ka, self.ka, self.stride, self.stride, self.padm, self.padm)
   end
   local para = nn.ParallelTable()

   if(self.cell2gate) then
       para:add(input2gate):add(output2gate):add(cell2gate)
   else
       para:add(input2gate):add(output2gate) 
   end
   gate:add(para)
   gate:add(nn.CAddTable())
   gate:add(nn.Sigmoid())

   return gate
end

function ConvLSTM:buildInputGate()
   self.inputGate = self:buildGate()
   return self.inputGate
end

function ConvLSTM:buildForgetGate()
   self.forgetGate = self:buildGate()
   return self.forgetGate
end

function ConvLSTM:buildcellGate()
   -- Input is : {input(t), output(t-1), cell(t-1)}, but we only need {input(t), output(t-1)}
   local hidden = nn.Sequential()
   hidden:add(nn.NarrowTable(1,2))
   local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   --local output2gate = nn.SpatialConvolution(self.outputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)

   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate)
   hidden:add(para)
   hidden:add(nn.CAddTable())
   hidden:add(nn.Tanh())
   self.cellGate = hidden
   return hidden
end

function ConvLSTM:buildcell()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.inputGate = self:buildInputGate() 
   self.forgetGate = self:buildForgetGate()
   self.cellGate = self:buildcellGate() -- cellGate is actually cell state

                                -- start --
   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()

   if(inputFlag) then
        concat:add(self.forgetGate):add(nn.SelectTable(3))
   else
        concat:add(self.forgetGate):add(nn.SelectTable(2))
   end

   forget:add(concat) -- {forgetGate},{selet cell activetion by selectable}
   forget:add(nn.CMulTable()) -- output = forgetGate} * cellActivetion
  --------------------------------

   -- input = inputGate{input(t), output(t-1), cell(t-1)} * cellGate{input(t), output(t-1), cell(t-1)}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.cellGate)
   input:add(concat2)
   input:add(nn.CMulTable())
   ---------------------------------


   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())

   self.cell = cell -- ct
   return cell

end   
   
function ConvLSTM:buildOutputGate()
   self.outputGate = self:buildGate()
   return self.outputGate
end

-- cell(t) = cell{input, output(t-1), cell(t-1)}
-- output(t) = outputGate{input, output(t-1)}*tanh(cell(t))
-- output of Model is table : {output(t), cell(t)} 
function ConvLSTM:buildModel()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.cell = self:buildcell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local model = nn.Sequential()

   if(self.inputFlag) then
       local concat = nn.ConcatTable()
       concat:add(nn.NarrowTable(1,2)) -- select {input, output(t-1)} 
             :add(self.cell)           -- cell gate {cell(t)}
       model:add(concat)               -- {{input(t), output(t-1)}, cell(t)}, 
   end

   -- output of concat is {{input(t), output(t-1)}, cell(t)}, 
   -- so flatten to {input(t), output(t-1), cell(t)}

   model:add(nn.FlattenTable())  -- as input to outputGate
   --- ============ partB of the model
   -------- output = 
   local cellAct = nn.Sequential() -- choose cell gate
   if(self.inputFlag) then
      cellAct:add(nn.SelectTable(3)) -- {intput gate}{}{cell gate}
   else
       cellAct:add(nn.SelectTable(2)) -- special case: input is {output(t-1), cell(t)}
   end
   -- notice: previous cell activation cell(t-1) is self.cell
   -- currenct cell stats if nn.cell:
   cellAct:add(nn.Tanh()) -- tanh
   local concat3 = nn.ConcatTable() -- concat outputgate and cell activation
   concat3:add(self.outputGate):add(cellAct)  -- {{outputGate}{cellActivation}}

   local output = nn.Sequential()
   output:add(concat3) 
   output:add(nn.CMulTable())                 -- {outputGate}*{cellActivation} = output
   ---------------------------------------------

   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()

   if(self.inputFlag) then
       concat4:add(output):add(nn.SelectTable(3))
   else
       concat4:add(output):add(nn.SelectTable(2))  -- special case: input is {output(t-1), cell(t)}
   end    

   model:add(concat4)  
   --[[ 
     after model add flattentable:

                  {input(t), output(t-1), cell(t)} 
concat&multi:       |:select(3)              |
  {outputGate} * {cellAct}                   | :selectTable(3)
              ||                             | :TanH()
     {     {output}           ,          {callAct}}  }
                             ||  
   ]]-- 

   return model
end

function ConvLSTM:updateOutput(input)
   local prevOutput, prevCell
   
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      prevCell = self.userPrevCell or self.zeroTensor
      if self.batchSize then
         self.zeroTensor:resize(self.batchSize,self.outputSize,input:size(3),input:size(4)):zero()
      else
         self.zeroTensor:resize(self.outputSize,input:size(2),input:size(3)):zero()
      end
   else
      -- previous output and memory of this module
      prevOutput = self.output
      prevCell   = self.cell
   end
      
   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end
   
   self.outputs[self.step] = output
   self.cells[self.step] = cell
   
   self.output = output
   self.cell = cell
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end

function ConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[2].modules[1].bias:fill(oBias)
  --self.inputGate.modules[2].modules[2].bias:fill(oBias)
  self.outputGate.modules[2].modules[1].bias:fill(oBias)
  --self.outputGate.modules[2].modules[2].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  --self.cellGate.modules[2].modules[2].bias:fill(oBias)
  self.forgetGate.modules[2].modules[1].bias:fill(fBias)
  --self.forgetGate.modules[2].modules[2].bias:fill(fBias)
end
