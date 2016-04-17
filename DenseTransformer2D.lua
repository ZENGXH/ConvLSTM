require 'nn'

local AGGOF, parent = torch.class('nn.AffineGridGeneratorOpticalFlow2D', 'nn.Module')

--[[
   AffineGridGeneratorOpticalFlow(height, width) :
   AffineGridGeneratorOpticalFlow:updateOutput(opticalFlowMap)
   AffineGridGeneratorOpticalFlow:updateGradInput(opticalFlowMap, gradGrids)

   AffineGridGeneratorOpticalFlow will take height x width x 2 x 3 an affine transform map (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will generate the next frame in the sequence according to the optical
   flow transform map.

   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function AGGOF:__init(height, width)
   parent.__init(self)
   assert(height > 1)
   assert(width > 1)
   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(2, height, width, opt.batchSize)

   for i=1, self.height do
      self.baseGrid:select(1,1):select(1,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1, self.width do
      self.baseGrid:select(1,2):select(2,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
   
   --self.baseGrid:select(1,3):fill(1)
end

function AGGOF:updateOutput(opticalFlowMap)
   -- ######! todo: input:  (batchSize, 2, H, W)
   -- input expected: (2, h, w, opt.batchSize)
--   print('opticalFlowMap')
--   print(opticalFlowMap:size())
   assert(opticalFlowMap:nDimension()==4 or opticalFlowMap:nDimension()==3
          and opticalFlowMap:size(1)== 2 , 'please input a valid transform map ')
   if opticalFlowMap:nDimension()==3 then 
      opticalFlowMap:resize(2, self.height, self.width, opt.batchSize)
   end
   -- need to scale the opticalFlowMap
   
   self.output:resize(2, self.height, self.width, opt.batchSize):zero()
--   print(self.baseGrid:size())
--   print(opticalFlowMap:size())
   self.output = torch.add(self.baseGrid, opticalFlowMap)
   
   return self.output
end

function AGGOF:updateGradInput(opticalFlowMap, gradGrid)
   self.gradInput:resizeAs(opticalFlowMap):zero()
   self.gradInput:copy(gradGrid)
   return self.gradInput
end
