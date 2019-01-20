### This is code for school project for the course 'Object Recognition' of Master MVA.

This is an adapted code based on the code for the paper *Dynamic Few-Shot Visual Learning without Forgetting*
source: https://github.com/gidariss/FewShotWithoutForgetting

### Requirements
It was developed and tested with pytorch version 0.2.0_4


### Training and evaluating our model on Mini-ImageNet.

**(1)** train a cosine-similarity based classifier and a feature extractor with 128 feature channels ) run the following command:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifier
```

**(2)** train the few-shot classification weight generator with attenition based weight inference) run the following commands:
```
# Training the model for the 1-shot case on the training set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1
# Training the model for the 5-shot case on the training set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5
```

**(3)** evaluate the above models run the following commands:
```
# Evaluating the model for the 1-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1 --testset
# Evaluating the model for the 5-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 --testset
```

**(4)** In order to train and evaluate our approach with different type of feature extractors (e.g., Conv128,ResNet10) change configuration, for example:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNet10CosineClassifier

CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNet10CosineClassifierGenWeightAttN1
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_ResNet10CosineClassifierGenWeightAttN1 --testset

CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNet10CosineClassifierGenWeightAttN5
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_ResNet10CosineClassifierGenWeightAttN5 --testset


In order to control the setting of data augmentation, just change the variable 'do_not_use_random_transf' in 'dataloader.py'