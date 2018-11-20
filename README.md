## PERSON RE-IDENTIFICATION PROJECT CSCE625

Trained Models folder contains the following:
1. ft_ResNet50		- Model trained using Market1501
2. ft_ResNetMixed	- Model trained using Market1501 and Duke MTMC
3. ft_ResNetBackRandom - Model trained using background subtracted Market1501
~~3. ft_ResNetBasic	- Model trained using Market1501 without flipping and cropping augmentations~~

## VALIDATION SET
Following are the results for  the validation dataset(.mat files of results are present in ./valSet/ directory):
1. ft_ResNetMixed- top1: 0.9067, top5: 0.9733, top10: 0.9867, mAP: 0.7969
2. ft_ResNetBackRandom - top1: 0.8333, top5: 0.9267, top10: 0.9400, mAP: 0.6902

## TEAM MEMBERS:
SARWESH KRISHNAN,
JOHN MATHAI.
