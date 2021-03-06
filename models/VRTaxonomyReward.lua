------------------------------------------------------------------------
--[[ VRTaxonomyReward ]]--
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction(one-hot encoded), baseline reward}
-- Reward is 5 for exact, 1 for similar, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
------------------------------------------------------------------------
local VRTaxonomyReward, parent = torch.class("nn.VRTaxonomyReward", "nn.Criterion")

function VRTaxonomyReward:__init(module, taxonomy, scale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
   self.taxonomy= taxonomy
end

function VRTaxonomyReward:updateOutput(inputTable, target)
   assert(torch.type(inputTable) == 'table')
   local input = self:toBatch(inputTable[1], 1)
   self._maxVal = self._maxVal or input.new()
   self._maxIdx = self._maxIdx or torch.type(input) == 'torch.CudaTensor' and torch.CudaLongTensor() or torch.LongTensor()

   -- max class value is class prediction
   self._maxVal:max(self._maxIdx, input, 2)
   if torch.type(self._maxIdx) ~= torch.type(target) then
      self._target = self._target or self._maxIdx.new()
      self._target:resize(target:size()):copy(target)
      target = self._target
   end

   -- reward = scale when correctly classified
   self.reward = self.reward or self._maxVal.new()
   self.reward:resize(self._maxVal:size(1)):zero()

   for i=1,input:size(1) do
      local taxonomy = self.taxonomy
      num = self.taxonomy:family_distance(self._maxIdx[i][1],target[i])
      if( num == 0 ) then self.reward[i]=10
      elseif( num == 2 ) then self.reward[i]=1
      else self.reward[i]=0
      end
   end

   self.reward:mul(self.scale)

   -- loss = -sum(reward)
   self.output = -self.reward:sum()
   --self.output = self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input:size(1)
   end
   return self.output
end

function VRTaxonomyReward:updateGradInput(inputTable, target)
   local input = self:toBatch(inputTable[1], 1)
   local baseline = self:toBatch(inputTable[2], 1)

   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   -- broadcast reward to modules
   self.module:reinforce(self.vrReward)

   -- zero gradInput (this criterion has no gradInput for class pred)
   self.gradInput[1]:resizeAs(input):zero()
   self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)

   -- learn the baseline reward
   self.criterion:forward(baseline, self.reward)
   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
   self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
   return self.gradInput
end

function VRTaxonomyReward:type(type)
   self._maxVal = nil
   self._maxIdx = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
