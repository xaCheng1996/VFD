# Released code for VFD
This is the release code for CVPR2022 paper ["Voice-Face Homogeneity Tells Deepfake"](https://arxiv.org/abs/2203.02195).

Part of the framework is borrowed from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

**Notes:** We only give a small batch of training and testing data, so some numerical modifications have been made to the dataset processing function to fit the small data. We will release the full data in a future official version.

## Model Architecture
The critical contribution of this paper is to determine the authenticity of videos cross deepfake datasets via the matching view of voices and faces. Except for the Voxceleb2 (which is difficult to access by now), you can employ any generic visual-audio datasets as training sets and test the model in deepfake datasets. We regard it as a fair comparison.

We applied the Transformer as the feature extractor to process the voice and face input. The ablation experiments show that these extractors will achieve SOTA results. However, we welcome any modifications to the feature extractors for efficiency or scalability as long as a clear statement of the model structure in the paper.

## Data Preprocess
For boosting the I/O speed, we have preprocessed the videos and audios in dataset, and stored them with the format showed in ./Dataset.
In paticular, for the videos (take voxceleb2 as example), we extract 1 frame in every video to represent the video. These frames are stored in the 
```
VFD/Dataset/Voxceleb2/face/id_number/video_name/frame_id
```
For example,
```
VFD/Dataset/Voxceleb2/face/id00015/JF-4trZP6fE/00182.jpg
```
Similarly, for the audios, we extract the Melspectrogram as representation with following code,
```
y, sr = librosa.load(audio_path, duration=3, sr=16000)
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=160, n_mels=512)
Mel_out = librosa.power_to_db(mel_spect, ref=np.max)
```
and the Mel_out is stored as .npy,
```
VFD/Dataset/Voxceleb2/voice/**id_number**_**video_name**_**frame_id**.npy
```
For example,
```
VFD/Dataset/Voxceleb2/voice/id00015_3X9uaIs66A0_00022.npy
```
## Quick Start
Train:

```
python train_DF.py --dataroot ./Dataset/Voxceleb2 --dataset_mode Vox_image --model DFD --no_flip --name experiment_name --serial_batches
```

Test (on DFDC):

```
python test_DF.py --dataroot ./Dataset/DFDC --dataset_mode DFDC --model DFD --no_flip --name experiment_name
```

## Note
If you have any problem when reading the paper or reproducing the code, please feel free to commit issue or contact us (E-mail: xacheng1996@gmail.com).
