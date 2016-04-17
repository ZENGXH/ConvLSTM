require 'nn'
--require 'cunn'
dofile '../PartialConnected.lua'
local backend_name = 'nn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end
  
local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 5, 5, 1, 1, 2,2))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(backend.ReLU(true))
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling
local AvePooling = nn.SpatialAveragePooling

ConvBNReLU(3,32)
--:add(nn.Dropout(0.3))
--ConvBNReLU(32,32)
vgg:add(MaxPooling(2,2,2,2)) -- 16

ConvBNReLU(32,32)
--:add(nn.Dropout(0.4))
--ConvBNReLU(128,128)
vgg:add(AvePooling(2,2,2,2))

ConvBNReLU(32,64)
vgg:add(AvePooling(2,2,2,2)) -- 4 
vgg:add(nn.SpatialConvolution(64, 120, 4, 4, 1, 1, 0, 0))
-- output: 100, 500, 1, 1
vgg:add(nn.PartialConnected(opt.batchSize, 120, 20))
-- input: 500, outputsize 20, ie 25 model, 


vgg:add(nn.SpatialConvolution(20, 10, 1, 1, 1, 1, 0, 0))
--:add(nn.Dropout(0.4))
--ConvBNReLU(256,256):add(nn.Dropout(0.4))
--ConvBNReLU(256,256)
--vgg:add(MaxPooling(2,2,2,2):ceil())

--ConvBNReLU(256,512):add(nn.Dropout(0.4))
--ConvBNReLU(512,512):add(nn.Dropout(0.4))
--ConvBNReLU(512,512)
--vgg:add(MaxPooling(2,2,2,2):ceil())

-- In the last block of convolutions the inputs are smaller than
-- the kernels and cudnn doesn't handle that, have to use cunn
backend = nn
-- ConvBNReLU(512,512):add(nn.Dropout(0.4))
-- ConvBNReLU(512,512):add(nn.Dropout(0.4))
-- ConvBNReLU(512,512)
-- vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(10))
--[[
classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
vgg:add(classifier)
--]]
-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'cudnn.SpatialConvolution'
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
