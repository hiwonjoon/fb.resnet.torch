th main.lua -reinforce true -taxonomy ./taxonomy-food.txt -rewardScale 1 -retrain ~/efs/model/torch-resnet-pretarined/resnet-34.t7 -data ~/efs/dataset/nv_dlcontest/ -resetClassifier true -nClasses 120 -gen ~/efs/torch-chkpoint/ -save ~/efs/torch-chkpoint/reinforce/ -LR 0.01 -optnet true
