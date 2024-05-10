## RME-GAN: A Learning Framework for Radio Map Estimation based on Conditional Generative Adversarial Network [ Under construction ]
### [Songyang Zhang], [Achintha Wijesinghe], and [Zhi Ding]

[[ArXiv Preprint]([https://arxiv.org/abs/2306.04321v1](https://arxiv.org/abs/2212.12817))]



### :page_with_curl: Abstract
Outdoor radio map estimation is an important tool for network planning and resource management in modern Internet of Things (IoT) and cellular systems. Radio map describes spatial signal strength distribution and provides network coverage information. A practical goal is to estimate fine-resolution radio maps from sparse radio strength measurements. However, non-uniformly positioned measurements and access obstacles can make it difficult for accurate radio map estimation (RME) and spectrum planning in many outdoor environments. In this work, we develop a two-phase learning framework for radio map estimation by integrating radio propagation model and designing a conditional generative adversarial network (cGAN). We first explore global information to extract the radio propagation patterns. We then focus on the local features to estimate the effect of shadowing on radio maps in order to train and optimize the cGAN. Our experimental results demonstrate the efficacy of the proposed framework for radio map estimation based on generative models from sparse observations in outdoor scenarios.



### :chart_with_upwards_trend: Main Results

<img src="fig1-Pagina-1.drawio.png"/>

###  Usage

#### Train RME-GAN

* Install the file `requirements.txt` and, separately, `conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch`.

* Run the following command:

```
python image_train.py --data_dir ./data --dataset_mode cityscapes --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True
--noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 35
--class_cond True --no_instance False 
```

For Cityscapes: `--dataset_mode cityscapes`, `--image_size 256`, `--num_classes 35`, `--class_cond True`, `--no_instance False`.

For COCO: `--dataset_mode coco`, `--image_size 256`, `--num_classes 183`, `--class_cond True`, `--no_instance False`.

For ADE20K: `--dataset_mode ade20k`, `--image_size 256`, `--num_classes 151`, `--class_cond True`, `--no_instance True`.

#### Sample from GESCO

* Train your own model or download our pretrained weights [here](https://drive.google.com/drive/folders/1MwDLhTM3MbhEm7z42zaJ0aITgPFxg-VT?usp=sharing).

* Run the following command:

```
python image_sample.py --data_dir "./data" --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 35 --class_cond True --no_instance False --batch_size 1 --num_samples 100 --model_path ./your_checkpoint_path.pt --results_path ./your_results_path --s 2 --one_hot_label True --snr your_snr_value --pool None --unet_model unet 
```

With the same dataset-specific hyperparameters, in addition to `--s` with is equal to `2` in Cityscapes and `2.5` for COCO and ADE20k.

Our code is based on [Radio Unet]([https://github.com/openai/guided-diffusion](https://github.com/RonLevie/RadioUNet))

#### Cite
Consider citing our paper, if it is found useful.

```
@ARTICLE{10130091,
  author={Zhang, Songyang and Wijesinghe, Achintha and Ding, Zhi},
  journal={IEEE Internet of Things Journal}, 
  title={RME-GAN: A Learning Framework for Radio Map Estimation Based on Conditional Generative Adversarial Network}, 
  year={2023},
  volume={10},
  number={20},
  pages={18016-18027},
  doi={10.1109/JIOT.2023.3278235}}
```

