# CCnet
ContrastControlNet: Official repository of "Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models"
In MICCAI 2024.


![method](docs/method.png)


## Getting Started

The [Duke Dataset](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) used in this study is available on [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903).


- [Scripts](src/bash/): Here you can find the scripts to start trainings and inference runs of AEKL, LDM, and ControlNet.
- [Code](src/python/): Here you can find the code for training, inference and evaluation of the models.
- [Config](configs/): Here you can find the configs for training, inference and evaluation of the models.
- [Data](data/): In the LDM_metadata.csv you can find the extracted metadata (e.g. containing scanner and contrast info) that can be input as text into the LDMs and ControlNet


## Fréchet Radiomics Distance (FRD)

You can find the FRD repository [here](https://github.com/RichardObi/frd-score). 

Let's get started and calculate the FRD with the code below: 

```
pip install frd-score

python -m frd_score path/to/dataset_A path/to/dataset_B
```


## Summary

![poster presentation](./poster.png)


## Reference
Please consider citing our work if you found it useful:
```bibtex
@inproceedings{osuala2024towards,
  title={Towards learning contrast kinetics with multi-condition latent diffusion models},
  author={Osuala, Richard and Lang, Daniel M and Verma, Preeti and Joshi, Smriti and Tsirikoglou, Apostolia and Skorupko, Grzegorz and Kushibar, Kaisar and Garrucho, Lidia and Pinaya, Walter HL and Diaz, Oliver and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={713--723},
  year={2024},
  organization={Springer}
}
```

## Acknowledgements
This repository borrows and extends the code from the [generative_brain_controlnet](https://github.com/Warvito/generative_brain_controlnet) repo. 
