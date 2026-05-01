LICENCE: Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/

# GazeD

This is the official PyTorch implementation of the paper *"[GazeD: Context-Aware Diffusion for Accurate 3D Gaze Estimation](https://arxiv.org/abs/2601.12948)"* (3DV 2026).

<p align="center">
  <img src="gif/gazed_sample.gif" />
</p>

GazeD is a diffusion-based model primarily for gaze estimation but can perform also pose estimation. This README provides instructions for setup, training, testing, and inference.




## 1. Setup

### Environment Setup 

You can find all the packages and dependencies in the environment.yml file. 
If you have conda, you can simply run

```
conda env create -f environment.yml

```
Otherwise you can refer to the requirements.txt file.

### Download the Webdatasets

Download the preprocessed datasets:

- [GFIE] (https://ailb-web.ing.unimore.it/publicfiles/gbData/GazeD/GazeD_WDS/GFIE.zip) 
- [GAFA] (https://ailb-web.ing.unimore.it/publicfiles/gbData/GazeD/GazeD_WDS/GAFA.zip) 
- [EgoExo](https://ailb-web.ing.unimore.it/publicfiles/gbData/GazeD/GazeD_WDS/EgoExo.zip)

You can store the webdatasets whethere you like, but you need to to specify the right path in the configuration files in the config folder in order to have the correct webdataset paths. Modify the voice dataset.root with your webdataset PATH.

### Download Pretrained Weights
Before starting testing, you need to download the pretrained weights [HERE](https://ailb-web.ing.unimore.it/publicfiles/gbData/GazeD/checkpoints.zip) 

Place PoseHRNET weights in the following folder:

```
checkpoint/posehrnet
```


## 3. Testing the Model

To test the model, download the corresponding pretrained weights and place them in the folder:

```
checkpoint/model_{dataset}_test
```
Then run:

```
python test.py --config config/{dataset}.yaml -c checkpoint --save_predictions -timesteps 20 -num_proposals 20\
                         --evaluate best_{dataset}.bin --dataset {dataset} 
```

### Explanation of Parameters:
- `--config config/{DATASET}.yaml` : Specifies the configuration file.
- `-c checkpoint` : Directory where model weights are stored.
- `--evaluate best_{DATASET}.bin` : Loads the best model weights for the specified dataset. This parameter is required. 
- `-timesteps 20` : Number of diffusion steps.
- `-num_proposals 20` : Number of hypotheses generated for the image.
- `--save_predictions` : Enables saving outputs inside ".npy" files in the predictions folder.
- `--dataset {DATASET}` : Specifies the dataset to be tested. Dataset can be GFIE,GAFA or EgoExo. Don't worry about CAPS, it should not be case sensitive.


## Citation

If you find our work useful for your project, please consider citing the paper:

```bibtex
@inproceedings{catalinigazed,
  title={GazeD: Context-Aware Diffusion for Accurate 3D Gaze Estimation},
  author={Catalini, Riccardo and Di Nucci, Davide and Borghi, Guido and Davoli, Davide and Garattoni, Lorenzo and Francesca, Gianpiero and Kawana, Yuki and Vezzani, Roberto},
  booktitle={Thirteenth International Conference on 3D Vision},
  year=2026
}
```
---


