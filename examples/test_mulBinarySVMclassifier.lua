require 'xlua'
require 'optim'
require 'nn'
require 'save_model'
require 'debugger'
dofile('./mulBinarySVMclassifier.lua')
require"logging.file"
date = ("2015_12_9")
local logger = logging.file("/home/xizeng/mylog/pcml2_train%s.log", "%Y-%m-%d")

logger:info("logging file of pcml2_training")
logger:debug("..")
logger:error("testing")
-- strat logging


-- require 'cunn'
dofile './cnnprovider.lua' -- import provider class definition

local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default 'logs'..date)      subdirectory to save logs
   -b,--batchSize             (default 40)          batch size
   -r,--learningRate          (default 0.00001)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 10)          epoch step
   --model                    (default cnnft_net)     model name
   --max_epoch                (default 60)           maximum number of iterations
]]

logger:info(opt)
print(opt)

last_loss = -10 -- init loss to a negetive value

--  torch_extra svmcriterion:
function L_i(outputs, targets)
  local delta = 1.0
  local loss = targets.new():resizeAs(targets):fill(0)
  for batch_i = 1, targets:size(1) do
    local score = outputs.new():resizeAs(outputs):copy(outputs)
    local correct_class_score = score[batch_i][targets[batch_i]]
    local D = 4
    local loss_i = 0.0
    for j = 1, D do
      if j ~= targets[batch_i] then
         loss_i = loss_i + math.max(0, score[batch_i][j] - correct_class_score + delta)
      end
    end
    loss[batch_i] = loss_i
  end
   print( loss/outputs:size(1))
   
   return loss/outputs:size(1)
end

function L_gradOutputs(outputs,targets, loss)
  local delta = 1.0
  local gradOutputs = outputs.new():resizeAs(outputs)
  local score = outputs.new():resizeAs(outputs):copy(outputs)
  for batch_i = 1, targets:size(1) do
    local loss_i = loss[batch_i]
    local outputs_i = outputs[batch_i]
    local gradOutputs_i = outputs_i.new():resizeAs(outputs_i) 
    local correct_class_score = score[batch_i][targets[batch_i]]
    --if yi = 2 = max(0, score_1 - score_yi + 1) + max(0, score_3 - score_yi + 1) + max(0, score_4 - score_yi + 1) 
    -- grad with respect to score_1 = 1 if first max != 0 
    local count_yi = 0
    for j = 1, 4 do
      if j ~= targets[batch_i] then
         if math.max(0, score[batch_i][j] - correct_class_score + delta) == 0 then
         	gradOutputs_i[j] = 0
         	count_yi = count_yi + 1
         else 
         	gradOutputs_i[j] = 1
         end
      end
    end
    gradOutputs_i[targets[batch_i]] = count_yi -- y_i
    gradOutputs[batch_i] = gradOutputs_i
  end
   return gradOutputs
end

--- criterion end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
-- DataAugmentation:
-- no filp for 36865 fts
-- model:add(nn.BatchFlip():float())
-- model:add(nn.Copy('torch:FloatTensor','torch:CudaTensor'):cuda()) --:cuda() make a variable CudaTensor
-- IMPORTANT: CAST MODEL TO FLOAT!

-- load model from file
model:add(dofile('models/'..opt.model..'.lua'))
-- model:add(nn.Euclidean(4,4))
model:add(nn.mulBinarySVMclassifier(36864))
-- model:add(nn.MulConstant(-1))
-- model:add(dofile('models/'..opt.model..'.lua'):cuda())
-- model:get(2).updateGradInput = function(input) return end
print(model)
model:float()
print(c.blue '==>' ..' loading data')
provider = torch.load '../train/cnnprovider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

provider_label = torch.load '../train/cnnprovider.t7'
provider_label.testData.data = provider_label.testData.data:float()

confusion = optim.ConfusionMatrix(4)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
-- criterion = nn.HingeEmbeddingCriterion(1)
criterion = nn.CrossEntropyCriterion()
-- criterion = nn.MultiLabelMarginCriterion() -- svm like loss -- output should be vector
-- criterion = nn.MultiMarginCriterion(1)


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  print('start training')
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  -- local targets = torch.CudaTensor(opt.batchSize)
  -- local targets = torch.FloatTensor(opt.batchSize)
  local targets = torch.FloatTensor(opt.batchSize) -- change to 4 since target now become rn vector
  -- local targets = torch.FloatTensor(opt.batchSize, 4) -- change to 4 since target now become rn vector
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()

  for t,v in ipairs(indices) do

   xlua.progress(t, #indices)
--   target_binary[torch.eq(targets, 2)] = -1
--   print(targets)
    --print(t)
    --print(v)
    local inputs = provider.trainData.data:index(1,v)
    
   -- targets:copy(provider.trainData.labels:index(1,v))
    targets:copy(provider_label.trainData.labels:index(1,v))
   --   print(targets)

    inputs = inputs:float()
--    print(inputs[{{1,10}}])

    targets = targets:float()

    --for i = 1,10 do print(inputs[1][i]) end

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      --print("debug: ")  
     -- print(torch.type(inputs))
      --print(torch.type(targets))
      model = model:float()
      local outputs = model:forward(torch.FloatTensor(inputs))
      targets = targets:float()
      outputs = outputs:float()

   target_binary = targets.new():resizeAs(targets):fill(-1)
   target_binary[torch.eq(targets, 4)] = 1
   targets_ori = targets.new():resizeAs(targets):copy(targets)
--   targets = target_binary


--      local ground_truth = torch.FloatTensor(opt.batchSize,4):fill(0)
      local ground_truth = targets
   --[[
      print(outputs:size())
      print('size of output')
      -- print prediction
    
      print(outputs)
   --]]
    --  print('size of output')
    --  print(outputs)
--      maxi,index = torch.max(outputs, 2)
--      print(index)
--      index:resize(opt.batchSize) -- resize to vector otherwise index[1] will not be num

--      for ind = 1,opt.batchSize do
--          ground_truth[ind][targets[ind]] = targets[ind]
--      end

--      print(prediction)
      -- print(torch.sum(index:eq(targets))/opt.Batchsize)
   
      -- if t%10 == 0 then print(outputs) end
      --print(targets)
--      print('done criterion loss')      
      criterion = criterion:float()
      -- local f = criterion:forward(outputs, targets)
--      print(ground_truth)
--      local f = L_i(outputs, ground_truth)
--      local df_do = L_gradOutputs(outputs,targets, f)
	
      local df_do = targets
      -- local f = criterion:forward(outputs, ground_truth)
--	print(f)
      -- local df_do = criterion:backward(outputs, targets)
      -- local df_do = criterion:backward(outputs, ground_truth)
--      print('start backward')
--      print(df_do)
      model:backward(inputs, df_do)
--	print('get f?')
--	print(f)
--    print(model.gradInput)
      -- confusion:batchAdd(outputs, targets)
--      print(targets_ori)
--      print(targets)
      output_tran = outputs.new():resizeAs(outputs):fill(4)
      output_tran[torch.eq(outputs, -1)] = 2
      ground_truth = targets.new():resizeAs(targets_ori):fill(4)
      ground_truth[torch.eq(targets, -1)] = 2
--      print(ground_truth)
--      print(output_tran)
	print(torch.sum(outputs))
      confusion:batchAdd(outputs, targets)
      
            confusion:updateValids() 
      --      print(confusion.totalValid * 100)
      if t%30 == 0
         then
            confusion:updateValids() 
            if last_loss > 0 then
                local cur_loss = confusion.totalValid * 100
                if cur_loss - last_loss > 5 then
                   local filename = paths.concat(opt.save, 'model'..date..'.net')
                   print('increase by 5!') 
                   saveModel(model,filename) -- save model call function in 'save_model.lua'
                end
            end
            last_loss = confusion.totalValid
            print(confusion.totalValid * 100)
            print(confusion)      
         end
      
      return f,gradParameters
    
    end

    optim.sgd(feval, parameters, optimState)
--    print(parameters)

  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  local trsize = 1200
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = opt.batchSize
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
            print(confusion)      
     data = torch.Tensor(trsize, 5408)
  print('p1')
  if testLogger then
    paths.mkdir(opt.save)
   -- testLogger:add{train_acc, confusion.totalValid * 100, confusion}
    testLogger:add{train_acc, confusion.totalValid*100}
   --testLogger:style{'-','-'}
   --  testLogger:plot()
print('p1')
    data = torch.Tensor(trsize,5408 )
    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

print('p1')
    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
print('p2')
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
--  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'final_model'..date..'.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
 --  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end


