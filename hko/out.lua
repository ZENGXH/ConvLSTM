{
  memorySizeH : 64
  seed : 1250
  output_nSeq : 15
  dataFile : /mnt/disk/ankur/Viorica/data/mnist/dataset_fly_64x64_lines_train.t7
  maxIter : 1000000
  display : false
  batchSize : 8
  inputSizeW : 50
  dir : outputs_hko
  v : false
  statInterval : 50
  save : true
  nFilters : 
    {
      1 : 1
    }
  stride : 1
  memorySizeW : 64
  padding : 1
  constrWeight : 
    {
      1 : 0
      2 : 1
      3 : 0.001
    }
  gradClip : 50
  dmax : 0.5
  displayInterval : 500
  eta : 0.0001
  transf : 2
  momentum : 0.9
  kernelSize : 3
  nSeq : 20
  etaDecay : 1e-05
  inputSizeH : 50
  input_nSeq : 5
  dmin : -0.5
  kernelSizeMemory : 3
  nFiltersMemory : 
    {
      1 : 4
      2 : 64
    }
}
intputsize	4	
 setup LSTM 	
Out[14]:
 setup LSTM 	
Out[14]:
 setup LSTM 	
Out[14]:
size of input: 	  8
  4
 50
 50
[torch.LongStorage of size 4]

 size of prevoutput: 	  8
 64
 50
 50
[torch.LongStorage of size 4]

size of prevCell	  8
 64
 50
 50
[torch.LongStorage of size 4]

Out[14]:
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.ConcatTable {
    input
      |`-> (1): nn.NarrowTable
      |`-> (2): nn.Sequential {
      |      [input -> (1) -> (2) -> output]
      |      (1): nn.ConcatTable {
      |        input
      |          |`-> (1): nn.Sequential {
      |          |      [input -> (1) -> (2) -> output]
      |          |      (1): nn.ConcatTable {
      |          |        input
      |          |          |`-> (1): nn.Sequential {
      |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |          |          |      (1): nn.NarrowTable
      |          |          |      (2): nn.ParallelTable {
      |          |          |        input
      |          |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
      |          |          |           ... -> output
      |          |          |      }
      |          |          |      (3): nn.CAddTable
      |          |          |      (4): nn.Sigmoid
      |          |          |    }
      |          |          |`-> (2): nn.SelectTable
      |          |           ... -> output
      |          |      }
      |          |      (2): nn.CMulTable
      |          |    }
      |          |`-> (2): nn.Sequential {
      |          |      [input -> (1) -> (2) -> output]
      |          |      (1): nn.ConcatTable {
      |          |        input
      |          |          |`-> (1): nn.Sequential {
      |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |          |          |      (1): nn.NarrowTable
      |          |          |      (2): nn.ParallelTable {
      |          |          |        input
      |          |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
      |          |          |           ... -> output
      |          |          |      }
      |          |          |      (3): nn.CAddTable
      |          |          |      (4): nn.Sigmoid
      |          |          |    }
      |          |          |`-> (2): nn.Sequential {
      |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |          |          |      (1): nn.NarrowTable
      |          |          |      (2): nn.ParallelTable {
      |          |          |        input
      |          |          |          |`-> (1): nn.SpatialConvolution(4 -> 64, 3x3, 1,1, 1,1)
      |          |          |          |`-> (2): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
      |          |          |           ... -> output
      |          |          |      }
      |          |          |      (3): nn.CAddTable
      |          |          |      (4): nn.Tanh
      |          |          |    }
      |          |           ... -> output
      |          |      }
      |          |      (2): nn.CMulTable
      |          |    }
      |           ... -> output
      |      }
      |      (2): nn.CAddTable
      |    }
       ... -> output
  }
  (2): nn.FlattenTable
  (3): nn.ConcatTable {
    input
      |`-> (1): nn.Sequential {
      |      [input -> (1) -> (2) -> output]
      |      (1): nn.ConcatTable {
      |        input
      |          |`-> (1): nn.Sequential {
      |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |          |      (1): nn.NarrowTable
      |          |      (2): nn.ParallelTable {
      |          |        input
      |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
      |          |           ... -> output
      |          |      }
      |          |      (3): nn.CAddTable
      |          |      (4): nn.Sigmoid
      |          |    }
      |          |`-> (2): nn.Sequential {
      |          |      [input -> (1) -> (2) -> output]
      |          |      (1): nn.SelectTable
      |          |      (2): nn.Tanh
      |          |    }
      |           ... -> output
      |      }
      |      (2): nn.CMulTable
      |    }
      |`-> (2): nn.SelectTable
       ... -> output
  }
}
{
  gradInput : DoubleTensor - empty
  modules : 
    {
      
Out[14]:
1 : 
        nn.ConcatTable {
          input
            |`-> (1): nn.NarrowTable
            |`-> (2): nn.Sequential {
            |      [input -> (1) -> (2) -> output]
            |      (1): nn.ConcatTable {
            |        input
            |          |`-> (1): nn.Sequential {
            |          |      [input -> (1) -> (2) -> output]
            |          |      (1): nn.ConcatTable {
            |          |        input
            |          |          |`-> (1): nn.Sequential {
            |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
            |          |          |      (1): nn.NarrowTable
            |          |          |      (2): nn.ParallelTable {
            |          |          |        input
            |          |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
            |          |          |           ... -> output
            |          |          |      }
            |          |          |      (3): nn.CAddTable
            |          |          |      (4): nn.Sigmoid
            |          |          |    }
            |          |          |`-> (2): nn.SelectTable
            |          |           ... -> output
            |          |      }
            |          |      (2): nn.CMulTable
            |          |    }
            |          |`-> (2): nn.Sequential {
            |          |      [input -> (1) -> (2) -> output]
            |          |      (1): nn.ConcatTable {
            |          |        input
            |          |          |`-> (1): nn.Sequential {
            |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
            |          |          |      (1): nn.NarrowTable
            |          |          |      (2): nn.ParallelTable {
            |          |          |        input
            |          |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
            |          |          |           ... -> output
            |          |          |      }
            |          |          |      (3): nn.CAddTable
            |          |          |      (4): nn.Sigmoid
            |          |          |    }
            |          |          |`-> (2): nn.Sequential {
            |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
            |          |          |      (1): nn.NarrowTable
            |          |          |      (2): nn.ParallelTable {
            |          |          |        input
            |          |          |          |`-> (1): nn.SpatialConvolution(4 -> 64, 3x3, 1,1, 1,1)
            |          |          |          |`-> (2): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
            |          |          |           ... -> output
            |          |          |      }
            |          |          |      (3): nn.CAddTable
            |          |          |      (4): nn.Tanh
            |          |          |    }
            |          |           ... -> output
            |          |      }
            |          |      (2): nn.CMulTable
            |          |    }
            |           ... -> output
            |      }
            |      (2): nn.CAddTable
            |    }
             ... -> output
        }
        {
          gradInput : DoubleTensor - empty
          modules : 
            {
              1 : 
                nn.NarrowTable
                {
                  gradInput : table: 0x0ccd5a80
                  offset : 1
                  length : 2
                  output : table: 0x0ccd5a58
                }
              2 : 
                nn.Sequential {
                  [input -> (1) -> (2) -> output]
                  (1): nn.ConcatTable {
                    input
                      |`-> (1): nn.Sequential {
                      |      [input -> (1) -> (2) -> output]
                      |      (1): nn.ConcatTable {
                      |        input
                      |          |`-> (1): nn.Sequential {
                      |    
Out[14]:
      |      [input -> (1) -> (2) -> (3) -> (4) -> output]
                      |          |      (1): nn.NarrowTable
                      |          |      (2): nn.ParallelTable {
                      |          |        input
                      |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
                      |          |           ... -> output
                      |          |      }
                      |          |      (3): nn.CAddTable
                      |          |      (4): nn.Sigmoid
                      |          |    }
                      |          |`-> (2): nn.SelectTable
                      |           ... -> output
                      |      }
                      |      (2): nn.CMulTable
                      |    }
                      |`-> (2): nn.Sequential {
                      |      [input -> (1) -> (2) -> output]
                      |      (1): nn.ConcatTable {
                      |        input
                      |          |`-> (1): nn.Sequential {
                      |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
                      |          |      (1): nn.NarrowTable
                      |          |      (2): nn.ParallelTable {
                      |          |        input
                      |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
                      |          |           ... -> output
                      |          |      }
                      |          |      (3): nn.CAddTable
                      |          |      (4): nn.Sigmoid
                      |          |    }
                      |          |`-> (2): nn.Sequential {
                      |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
                      |          |      (1): nn.NarrowTable
                      |          |      (2): nn.ParallelTable {
                      |          |        input
                      |          |          |`-> (1): nn.SpatialConvolution(4 -> 64, 3x3, 1,1, 1,1)
                      |          |          |`-> (2): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
                      |          |           ... -> output
                      |          |      }
                      |          |      (3): nn.CAddTable
                      |          |      (4): nn.Tanh
                      |          |    }
                      |           ... -> output
                      |      }
                      |      (2): nn.CMulTable
                      |    }
                       ... -> output
                  }
                  (2): nn.CAddTable
                }
                {
                  gradInput : DoubleTensor - empty
                  modules : table: 0x0d518658
                  output : DoubleTensor - empty
                }
            }
          output : table: 0x0ccd5950
        }
      2 : 
        nn.FlattenTable
        {
          gradInput : table: 0x0ccd5b90
          input_map : table: 0x0ccd5b68
          output : table: 0x0ccd5b40
        }
      3 : 
        nn.ConcatTable {
          input
            |`-> (1): nn.Sequential {
            |      [input -> (1) -> (2) -> output]
            |      (1): nn.ConcatTable {
            |        input
            |          |`-> (1): nn.Sequential {
            |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
            |          |      (1): nn.NarrowTable
            |          |      (2): nn.ParallelTable {
            |          |        input
            |          |          |`-> (1): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
            |          |           ... -> output
            |          |      }
            |          |      (3): nn.CAddTable
            |          |      (4): nn.Sigmoid
            |          |    }
            |          |`-> (2): nn.Sequential {
            |          |      [input -> (1) -> (2) -> output]
            |          |      (1): nn.SelectTable
            |          |      (2): nn.Tanh
/Users/zengxiaohui/torch/install/share/lua/5.1/nn/THNN.lua:1077: Wrong number of input channels! Input has 4 channels, expected 64 at /Users/zengxiaohui/torch/extra/nn/lib/THNN/generic/SpatialConvolutionMM.c:89
stack traceback:
	[C]: in function 'v'
	/Users/zengxiaohui/torch/install/share/lua/5.1/nn/THNN.lua:1077: in function 'SpatialConvolutionMM_updateOutput'
	...ui/torch/install/share/lua/5.1/nn/SpatialConvolution.lua:104: in function 'updateOutput'
	...xiaohui/torch/install/share/lua/5.1/nn/ParallelTable.lua:12: in function 'updateOutput'
	...engxiaohui/torch/install/share/lua/5.1/nn/Sequential.lua:44: in function 'updateOutput'
	...ngxiaohui/torch/install/share/lua/5.1/nn/ConcatTable.lua:11: in function 'updateOutput'
	...engxiaohui/torch/install/share/lua/5.1/nn/Sequential.lua:44: in function 'updateOutput'
	...ngxiaohui/torch/install/share/lua/5.1/nn/ConcatTable.lua:11: in function 'updateOutput'
	...engxiaohui/torch/install/share/lua/5.1/nn/Sequential.lua:44: in function 'updateOutput'
	...ngxiaohui/torch/install/share/lua/5.1/nn/ConcatTable.lua:11: in function 'updateOutput'
	...engxiaohui/torch/install/share/lua/5.1/nn/Sequential.lua:44: in function 'updateOutput'
	...
	[C]: in function 'xpcall'
	.../zengxiaohui/torch/install/share/lua/5.1/itorch/main.lua:209: in function <.../zengxiaohui/torch/install/share/lua/5.1/itorch/main.lua:173>
	.../zengxiaohui/torch/install/share/lua/5.1/lzmq/poller.lua:75: in function 'poll'
	...ngxiaohui/torch/install/share/lua/5.1/lzmq/impl/loop.lua:307: in function 'poll'
	...ngxiaohui/torch/install/share/lua/5.1/lzmq/impl/loop.lua:325: in function 'sleep_ex'
	...ngxiaohui/torch/install/share/lua/5.1/lzmq/impl/loop.lua:370: in function 'start'
	.../zengxiaohui/torch/install/share/lua/5.1/itorch/main.lua:381: in main chunk
	[C]: in function 'require'
	(command line):1: in main chunk
	[C]: at 0x010c9cbbc0
