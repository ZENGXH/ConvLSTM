require 'torch'
require 'nn'
function saveModel(model,modelname, iter)
    --require 'torch'
    --require 'nn'
	overlap = false
	if overlap then
		iter = 0
	end
	local dir = 'image/'

	if true then
		local emptymodel = model:clone('weight', 'bias'):float()
		local filename = dir..modelname..'iter-'..tostring(iter)..'.bin' 
		torch.save(filename, emptymodel)
		print('current model is saved as')
		print(filename)
    end
end
--[[
function saveModel(encoder_0, encoder_1, decoder_2, decoder_3, convForward_4, iter)
    --require 'torch'
    --require 'nn'
	overlap = true
	if overlap then
		iter = 0
	end
	local dir = 'image/'

	if true then
		local emptyencoder_0 = encoder_0:clone('weight', 'bias'):float()
		local filename = dir..'iter-'..tostring(iter)..'-encoder_0.bin' 
		torch.save(filename, emptyencoder_0)
		print('current model is saved as')
		print(filename)
    end
    if true then
		local emptyencoder_1 = encoder_1:clone('weight', 'bias'):float()
		filename = dir..'iter-'..tostring(iter)..'-encoder_1.bin' 
		torch.save(filename, emptyencoder_1)
		print('current model is saved as')
		print(filename)
	end

	if true then
		local emptydecoder_2 = decoder_2:clone('weight', 'bias'):float()
		filename = dir..'iter-'..tostring(iter)..'-decoder_2.bin' 
		torch.save(filename, emptydecoder_2)
		print('current model is saved as')
		print(filename)
	end

	if true then
		local emptydecoder_3 = decoder_3:clone('weight', 'bias'):float()
		filename = dir..'iter-'..tostring(iter)..'-decoder_3.bin' 
		torch.save(filename, emptydecoder_3)
		print('current model is saved as')
		print(filename)
	end

	if true then
		local emptyconv4 = convForward_4:clone('weight', 'bias'):float()
		filename = dir..'iter-'..tostring(iter)..'-convForward_4.bin' 
		torch.save(filename, emptyconv4)
		print('current model is saved as')
		print(filename)
	end
	
end
]]
