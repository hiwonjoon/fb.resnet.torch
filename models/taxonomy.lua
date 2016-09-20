local Taxonomy = torch.class('Taxonomy')

--local function findClasses(dir)
--   local dirs = paths.dir(dir)
--   table.sort(dirs)
--
--   local classList = {}
--   local classToIdx = {}
--   for _ ,class in ipairs(dirs) do
--      if not classToIdx[class] and class ~= '.' and class ~= '..' then
--         table.insert(classList, class)
--         classToIdx[class] = #classList
--      end
--   end
--
--   -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
--   return classList, classToIdx
--end

function Taxonomy:__init(opt)
   self.opt = opt
   -- Currently, only two level tree is supported.
   -- TODO!
    file = io.open(opt.taxonomy) --opt.taxonomy)
    local numClasses = tonumber(file:read())
    local family_num = torch.IntTensor(numClasses,numClasses):fill(4)
    for line in file:lines() do
        bros = line:split(",")
        for _,x in pairs(bros) do
            for _,y in pairs(bros) do
                i = tonumber(x); j = tonumber(y)
                if(i==j) then family_num[i][j] = 0
                else family_num[i][j] = 2 end
            end
        end
    end
    file:close()
    self.family_num = family_num
end

function Taxonomy:family_distance(l1,l2)
    return self.family_num[l1][l2]
end

