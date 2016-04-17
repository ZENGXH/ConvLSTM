dofile 'BinarySVMclassifier.lua'
a = nn.BinarySVMclassifier(3)
input = torch.Tensor{{2,2,2}, {3,3,3}, {5,5,5}}
target = torch.Tensor{-1, 1, 1}


for i = 1, 3 do
output = a:forward(input)
print('output')
print(output)
a:backward(input,target )
b = a:parameters()
print(b[1])


output = a:forward(input)
print('output')
print(output)
a:parameters()
--print(f)
a:backward(input,target )
end
--print(a.gardInput)
--print(a.gardWeight)
