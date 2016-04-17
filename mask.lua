require 'torch'
require 'image'

function createMask(height, width, offsetx, offsety, stride) 
   local offsetx = offsetx or 0
   local offsety = offsety or 0
   local stride = stride or 2
   local mask = torch.ByteTensor(height, width):fill(0)
   for i = 1, height/stride do
       for j = 1, width/stride do 
           mask[stride * i - 1 + offsety][stride * j - 1 + offsetx] = 1
       end
    end
    return mask
end

function applyMask(mask, image, height, width, stride)
    local img = image:clone():float()
    local subImg = torch.Tensor(height / stride, width / stride):fill(0):float()
    local newimg = img:maskedSelect(mask):clone():resizeAs(subImg)
    return newimg
end

function recoverMask(originImg, mask, subimage, height, width, stride)
    local subimg = subimage:clone():float()
    -- local img = torch.Tensor(height, width):fill(0):float()
    originImg:maskedCopy(mask, subimg)
    return originImg
end

function quick4Mask3Dforward(originImg)
   assert(torch.isTensor(originImg))
   assert(originImg:nDimension() == 2 or originImg:nDimension() == 3)
   if originImg:nDimension() == 2 then
       image:resize(1, originImg:size(1), originImg:size(2))
   end
   local stride = 2
   local height = originImg:size(2)
   local width = originImg:size(3)
   local imageFakeDepth = 4 -- not really depth, can be regard as batchsize
   local newImage = torch.Tensor(imageFakeDepth, height/stride, width/stride):float()
   local i = 1
   for offsetx = 0,1 do
       for offsety = 0,1 do
           local mask = createMask(height, width, offsetx, offsety, stride)
           local newSubImage = applyMask(mask, originImg, height, width, stride)
           newImage[i] = newSubImage
           i = i + 1
       end
    end
  return newImage
end

function quick4Mask3Dbackward(subImage)
   assert( subImage:nDimension() == 3)
   -- sub image in size(4, height, width)
   local stride = 2
   local height = subImage:size(2) * stride
   local width = subImage:size(3) * stride
   local imageFakeDepth = subImage:size(1) -- not really depth, can be regard as batchsize
   local originImg = torch.Tensor(1, height, width):float()
   local i = 1
   for offsetx = 0,1 do
       for offsety = 0,1 do
           local sub = subImage[i]
           local mask = createMask(height, width, offsetx, offsety, stride)
           recoverMask(originImg, mask, sub, height, width, stride)
           i = i + 1
       end
    end
  return originImg
end


function test()
   print('please test with qlua compiler')
   local im = image.lena()
   im = image.rgb2y(im)
   local newImg = quick4Mask3Dforward(im)
   for i = 1, 4 do
      local single_subImg_fordisplay = newImg:select(1, i) 
      image.display(single_subImg_fordisplay)
   end

   print('shffule back: ')
   local originImg = quick4Mask3Dbackward(newImg) 
   image.display(originImg)
   print('check if pass test')

end
