require 'nn'
require 'rnn'
-- require 'cunn'
require 'DenseTransformer2D'
-- require 'SmoothHuberPenalty'


-- if affine transform, opt,transf must be 2

flow = nn.Sequential()
local pad = torch.floor(opt.kernelSizeFlow / 2)
local conv = nn.Sequential()
conv:add(nn.SpatialConvolution(opt.nFiltersMemory[2], opt.transfBetween, 
								opt.kernelSizeFlow, 
								opt.kernelSizeFlow, 1, 1, pad, pad))
-- print(conv.modules)
local conv_new = require('weight-init')(conv, 'xavier')
flow:add(conv_new) -- regression layer 1

local conv2 = nn.Sequential()
conv2:add(nn.SpatialConvolution(opt.transfBetween, opt.transf, 
								opt.kernelSizeFlow, opt.kernelSizeFlow,
								1, 1, 
								pad, pad))

local conv_new2 = require('weight-init')(conv2, 'xavier')
flow:add(conv_new2) -- regression layer 2

-- add an extra convolutional layer useful to hard code initial flow map with 0
local conv0 = nn.SpatialConvolution(opt.transf, opt.transf, 1, 1, 1, 1)
conv0.weight:fill(0)
conv0.bias:fill(0)
flow:add(conv0) -- :add(nn.Reshape(opt.nFiltersMemory[2], opt.memorySizeH, opt.memorySizeW))

-- need to rescale the optical flow vectors since the sampler considers the image size between [-1,1]
-- flow:add(nn.SplitTable(2))

-- concat = nn.ConcatTable()
-- subBatch = nn.Sequential()
local b1 = nn.Sequential()
local m1 = nn.Mul()
m1.weight = torch.Tensor{ 2 / opt.memorySizeH }
-- m1.weight = torch.Tensor{2/50}
b1:add(m1):add(nn.View(opt.batchSize , 2, opt.memorySizeH, opt.memorySizeW)) 
-- reshape into (1, 32, 32)
--[[
local b2 = nn.Sequential()
local m2 = nn.Mul()
m2.weight = torch.Tensor{ 2 / opt.memorySizeW } 
b2:add(m2):add(nn.View(opt.batchSize, 1, opt.memorySizeH, opt.memorySizeW))
-- all input multiply with 2/memorySizeW, 2/32
]]
-- local para = nn.ConcatTable()
-- para:add(b1)-- :add(b2)

-- flow:add(para)
flow:add(b1)

-- flow:add(nn.JoinTable(2)) -- output: (batchSize, 2, H, W)

-- clamp optical flow values to make sure they stay within image limits
flow:add(nn.Clamp(opt.dmin, opt.dmax))

-- flow:add(para)
-- flow:add(nn.JoinTable(1))

-- clamp optical flow values to make sure they stay within image limits
-- flow:add(nn.Clamp(opt.dmin, opt.dmax))

-- next module does not modify its input, 
-- only accumulates penalty in backprop pass to penalise non-smoothness 
-- flow:add(nn.SmoothHuberPenalty(opt.transf,opt.constrWeight[3]))
flow:add(nn.Transpose({1, 2}, {2, 3}, {3, 4}))  -- bdhw -> dhwb
   -- ######! todo: input:  (tempBatchSize = 4, 2, H, W) -> (W, 2, H, b) -> (2, W, H, tempBatchSize = 4)
   -- input expected: (2, h, w, tempBatchSize)
flow:add(nn.AffineGridGeneratorOpticalFlow2D(opt.memorySizeH,opt.memorySizeW)) -- apply transformations to obtain new grid
   -- output: (2, self.height, self.width, opt.batchSize)
   -- (b, h, w, 2)
-- transpose to prepare grid in bhwd format for sampler
flow:add(nn.Transpose({1, 4}))  -- dhwb -> BilinearSamplerBHWD
        --:add(nn.Reshape(opt.memorySizeH, opt.memorySizeW, opt.transf, 1))


--[[
th> net:add(flow)
nn.Sequential {
  [input -> (1) -> output]
  (1): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> output]
    (1): nn.Sequential {
      [input -> (1) -> output]
      (1): nn.SpatialConvolution(45 -> 2, 15x15, 1,1, 7,7)
    }
    (2): nn.Sequential {
      [input -> (1) -> output]
      (1): nn.SpatialConvolution(2 -> 2, 15x15, 1,1, 7,7)
    }
    (3): nn.SpatialConvolution(2 -> 2, 1x1)
    (4): nn.SplitTable
    (5): nn.ParallelTable {
      input
        |`-> (1): nn.Sequential {
        |      [input -> (1) -> (2) -> output]
        |      (1): nn.Mul: multiply with 2/memorySize H = 2/32 = 1/16
        |      (2): nn.Reshape(1x32x32)
        |    }
        |`-> (2): nn.Sequential {
        |      [input -> (1) -> (2) -> output]
        |      (1): nn.Mul: multiply with memorySize W = 2/32 = 1/16
        |      (2): nn.Reshape(1x32x32)
        |    }
         ... -> output
    }
    (6): nn.JoinTable: output 2x32x32 
    (7): nn.Clamp:  threhold the image
    (8): nn.AffineGridGeneratorOpticalFlow2D: add with the base grid
    (9): nn.Transpose
  }
}




]]