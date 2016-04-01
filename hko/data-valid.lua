require 'image'
local data_verbose = false

function getdataSeq_valid(data_path)
   -- local data = torch.DiskFile(datafile,'r'):readObject()
   --  local data_path = 
   -- data size (totalInstances or nsamples=2000?, sequence_length=20, 1, 64, 64)
   local datasetSeq ={}
   -- data = data:float()/255.0 -- to range(0, 1)

   --------------- configuration: -----------------
--   local std = std or 0.2
   local nsamples = 2037 -- data:size(1)
   local nseq  = 20 -- data:size(2)
   local nrows = 100 -- data:size(4)
   local ncols = nrows -- data:size(5)
   local nbatch = opt.batchSize
   print (nsamples .. ' ' .. nseq .. ' ' .. nrows .. ' ' .. ncols)

   ------------- read the powerful txt file! ------
   local fileList = {}
   f = io.open(data_path..'validseq.txt', 'r')
   local id = 1
   for line in f:lines() do
      fileList[id] = line
      id = id + 1
   end
   assert(table.getn(fileList) == nseq * nsamples)

   function datasetSeq:size()
      return nsamples
   end

   function datasetSeq:selectSeq()
      local imageok = false
      if simdata_verbose then
         print('selectSeq')
      end

      while not imageok do
         local input_batch = torch.Tensor(nbatch, nseq, 1, nrows, ncols)
 
         for batch_ind = 1, nbatch do
            local i = math.ceil(torch.uniform(1e-12,nsamples)) 
            -- choose an index in range{1,.. nsamples}
            -- image index
            -- read the 20 frames starting from i
            for k = 1, nseq do
               input_batch[batch_ind][k] = image.load(data_path..'data/'..fileList[(i-1)*nseq + k])
               -- local loa = image.load(data_path..'img'..tostring((i-1)*5 + k)..'.png')
               -- print(loa:size())
               -- print(input_batch[batch_ind][k]:size())
               --assert(loa:size() == input_batch[batch_ind][k]:size(), 
               --   "is"..tostring(loa:size())..tostring(input_batch[batch_ind][k]:size()))
               --input_batch[batch_ind][k] = loa

            end
         end

         -- local im = data:select(1,i)
         -- along dimension 1 of data, select the ith slice of it
         return input_batch,i
      end
   end

   dsample = torch.Tensor(nbatch, nseq, 1, nrows, ncols)
   -- dsample(20, 1, 64, 64)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample, i = self:selectSeq()
                                       dsample:copy(sample)
                                       return {dsample,i}
                                    end})
   return datasetSeq
end
