local nn = require 'nn'
require 'cunn'
require 'rnn'


local resnet = require 'resnet'

HIDDEN_FEATURES = 256

local function createModel(opt)
    base_model = resnet(opt)
    local orig = base_model:get(#base_model.modules)
    assert(torch.type(orig) == 'nn.Linear',
       'expected last layer to be fully connected')

    local linear = nn.Linear(orig.weight:size(2), HIDDEN_FEATURES)
    linear.bias:zero()

    base_model:remove(#base_model.modules)
    base_model:add(linear:cuda())

    local lstm = nn.Sequential()
        :add(base_model)
        :add(nn.FastLSTM(HIDDEN_FEATURES, HIDDEN_FEATURES):maskZero(1))

    local linear = nn.Linear(orig.weight:size(2), HIDDEN_FEATURES)

    lstm:add(linear:cuda())
    lstm:add(nn.Softmax())
    lstm:add(nn.ReinforceCategorical(stochastic=True)) --pick based on probability

   -- add the baseline reward predictor
   seq = nn.Sequential()
   seq:add(nn.Constant(1,1))
   seq:add(nn.Add(1))
   concat = nn.ConcatTable():add(nn.Identity()):add(seq)
   concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

   -- Not finished. Let's do other things first.
   -- basic idea is, during the first learn, let them wrong!
   -- 그러니깐, 1-iteration에서는 2번쨰 트리레벨 까지만 맞춰도 그냥 맞춘 것으로 취급해
   -- tree depth가M이면, M-iteration에서는 완전히 맞춰야 reward를 받음.









