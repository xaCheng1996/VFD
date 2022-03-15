# Released code for VFD
This is the release code for CVPR2022 paper ["Voice-Face Homogeneity Tells Deepfake"](https://arxiv.org/abs/2203.02195).

Part of the framework is borrowed from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

**Notes:** We only give a small batch of training and testing data, so some numerical modifications have been made to the dataset processing function to fit the small data. We will release the full data in a future official version.

## Model Architecture
The critical contribution of this paper is to determine the authenticity of videos cross deepfake datasets via the matching view of voices and faces. Except for the Voxceleb2 (which is difficult to access by now), you can employ any generic visual-audio datasets as training sets and test the model in deepfake datasets. We regard it as a fair comparison.

We applied the Transformer as the feature extractor to process the voice and face input. The ablation experiments show that these extractors will achieve SOTA results. However, we welcome any modifications to the feature extractors for efficiency or scalability as long as a clear statement of the model structure in the paper.

## Quick Start
Train:

```
python train_DF.py --dataroot ./Dataset/Voxceleb2 --dataset_mode Vox_image --model DFD --no_flip --name experiment_name --serial_batches
```

Test (on DFDC):

```
python test_DF.py --dataroot ./Dataset/DFDC --dataset_mode DFDC --model DFD --no_flip --name experiment_name

##Note
If you have any problem when reading the paper or reproducing the code, please feel free to commit issue or email us [E-mail](xacheng1996@gmail.com).
