-- based on noteds in http://cs231n.github.io/linear-classify/#softmax
--[[ FOR testing
  outputs = torch.rand(5,4)
  outputs[{{},{1}}]  = 1-- batch:5, class 3
  targets = torch.Tensor{1,1,1,1,1}
--]]

function L_i(outputs, targets)
  local delta = 1.0
  local loss = targets.new():resizeAs(targets):fill(0)
  for batch_i = 1, targets:size(1) do
    local score = outputs.new():resizeAs(outputs):copy(outputs)
    local correct_class_score = score[batch_i][targets[batch_i]]
    local D = 4 -- 4 CLASS
    local loss_i = 0.0
    for j = 1, D do
      if j ~= targets[batch_i] then
         loss_i = loss_i + math.max(0, score[batch_i][j] - correct_class_score + delta)
      end
    end
    loss[batch_i] = loss_i
  end
   return loss
end

function L_gradOutputs(outputs, loss)
  local gradOutputs = outputs.new():resizeAs(outputs)
  for batch_i = 1, targets:size(1) do
  	local loss_i = loss[batch_i]
  	local outputs_i = outputs[batch_i]
    local gradOutputs_i = outputs_i.new():resizeAs(outputs_i) 
    --if yi = 2 = max(0, score_1 - score_yi + 1) + max(0, score_3 - score_yi + 1) + max(0, score_4 - score_yi + 1) 
    -- grad with respect to score_1 = 1 if first max != 0 
    local count_yi = 0
    for j = 1, D do
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

