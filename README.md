## RME-GAN: A Learning Framework for Radio Map Estimation based on Conditional Generative Adversarial Network [ Under construction ]
### [Songyang Zhang](https://scholar.google.com/citations?user=cZI11G4AAAAJ&hl=en), [Achintha Wijesinghe](https://scholar.google.com/citations?hl=en&user=PGVNF-gAAAAJ), and [Zhi Ding](https://scholar.google.com/citations?user=G0oZBhUAAAAJ&hl=en)

[ArXiv Preprint](https://arxiv.org/abs/2212.12817)
[Paper](https://ieeexplore.ieee.org/document/10130091)



### Abstract
Outdoor radio map estimation is an important tool for network planning and resource management in modern Internet of Things (IoT) and cellular systems. Radio map describes spatial signal strength distribution and provides network coverage information. A practical goal is to estimate fine-resolution radio maps from sparse radio strength measurements. However, non-uniformly positioned measurements and access obstacles can make it difficult for accurate radio map estimation (RME) and spectrum planning in many outdoor environments. In this work, we develop a two-phase learning framework for radio map estimation by integrating radio propagation model and designing a conditional generative adversarial network (cGAN). We first explore global information to extract the radio propagation patterns. We then focus on the local features to estimate the effect of shadowing on radio maps in order to train and optimize the cGAN. Our experimental results demonstrate the efficacy of the proposed framework for radio map estimation based on generative models from sparse observations in outdoor scenarios.


###  Usage

#### Train RME-GAN

* We have used the PyTorch framework for the model implementation.

* Run the following command:

```
python3 train.py
```
* To change the different setups, change the file lib/loadersUNETCGAN_f.py

Our code is based on [Radio Unet](https://github.com/RonLevie/RadioUNet).

You may want to test on the [RadioMapSeer Dataset](https://radiomapseer.github.io/).

Another available dataset is [BRATLab Dataset](https://github.com/BRATLab-UCD/Radiomap-Data).

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

