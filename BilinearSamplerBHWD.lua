assert(nn.BilinearSamplerBHWD, "stnbhwd package not preloaded")

-- we overwrite the module of the same name found in the stnbhwd package
local BilinearSamplerBHWD, parent = nn.BilinearSamplerBHWD, nn.Module

--[[
   BilinearSamplerBHWD() :
   BilinearSamplerBHWD:updateOutput({inputImages, grids})
   BilinearSamplerBHWD:updateGradInput({inputImages, grids}, gradOutput)

   BilinearSamplerBHWD will perform bilinear sampling of the input images according to the
   normalized coordinates provided in the grid. Output will be of same size as the grids, 
   with as many features as the input images.

   - inputImages has to be in BHWD layout

   - grids have to be in BHWD layout, with dim(D)=2
   - grids contains, for each sample (first dim), the normalized coordinates of the output wrt the input sample
      - first coordinate is Y coordinate, second is X
      - normalized coordinates : (-1,-1) points to top left, (-1,1) points to top right
      - if the normalized coordinates fall outside of the image, then output will be filled with zeros
]]

function BilinearSamplerBHWD:__init()
   parent.__init(self)
   self.gradInput={}
end

function BilinearSamplerBHWD:check(input, gradOutput)
   local inputImages = input[1]
	local grids = input[2]
   --print('input image')
   --print(inputImages:size())
   assert(inputImages:nDimension()==4)
   assert(grids:nDimension()==4)
   assert(inputImages:size(1)==grids:size(1)) -- batch
   assert(grids:size(4)==2) -- coordinates

   if gradOutput then
      assert(grids:size(1)==gradOutput:size(1))
      assert(grids:size(2)==gradOutput:size(2))
      assert(grids:size(3)==gradOutput:size(3))
   end
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function BilinearSamplerBHWD:updateOutput(input)
	local _inputImages = input[1]
	local _grids = input[2]

   local inputImages, grids

   if _inputImages:nDimension()==3 then
      inputImages = addOuterDim(_inputImages)
      grids = addOuterDim(_grids)
   else

      inputImages = _inputImages
      grids = _grids

   end

   local input = {inputImages, grids}

   self:check(input)

   self.output:resize(inputImages:size(1), grids:size(2), grids:size(3), inputImages:size(4))	

	inputImages.nn.BilinearSamplerBHWD_updateOutput(self, inputImages, grids)

   if _inputImages:nDimension()==3 then
      self.output=self.output:select(1,1)
   end
	
   return self.output
end

function BilinearSamplerBHWD:updateGradInput(_input, _gradOutput)
	local _inputImages = _input[1]
	local _grids = _input[2]

   local inputImages, grids, gradOutput
   if _inputImages:nDimension()==3 then
      inputImages = addOuterDim(_inputImages)
      grids = addOuterDim(_grids)
      gradOutput = addOuterDim(_gradOutput)
   else
      inputImages = _inputImages
      grids = _grids
      gradOutput = _gradOutput
   end

   local input = {inputImages, grids}

   self:check(input, gradOutput)
	for i=1,#input do
	   self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i]):zero()
   end

   local gradInputImages = self.gradInput[1]
   local gradGrids = self.gradInput[2]

   inputImages.nn.BilinearSamplerBHWD_updateGradInput(self, inputImages, grids, gradInputImages, gradGrids, gradOutput)

   if _gradOutput:nDimension()==3 then
      self.gradInput[1]=self.gradInput[1]:select(1,1)
      self.gradInput[2]=self.gradInput[2]:select(1,1)
   end
   
   return self.gradInput
end

function BilinearSamplerBHWD:_updateOutput(inputImages, grids)
   local inputImages = inputImages -- (batchSizem height, width, depth)
   local grids = grids -- (batchSizem height, width, 2)
   local output = inputImages:new()

   local batchSize = inputImages:size(1)
   local inputImages_height = inputImages:size(2)
   local inputImages_width = inputImages:size(3)
   local output_height = inputImages_height
   local output_width = inputImages_width
   local inputImages_channels = inputImages:size(4)
--[[
  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[1];
  int output_strideWidth = output->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];
--]]
   for b = 1, batchSize do
      for yOut = 1, output_height do
         for xOut = 1, output_width do
            local yf = grids[b][yOut][xOut][1]
            local xf = grids[b][yOut][xOut][2]
            local xcoord = (xf + 1)/ 2 * (inputImages_width - 1) + 1; -- xf = -1 => xcoord = 1, xf = 1 => xcoord = inputImages_width
            local ycoord = (yf + 1)/ 2 * (inputImages_height - 1) + 1; 
         end
      end
   end


