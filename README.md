# Combining open and closed vocabulary segmentation with Gaussian Splatting

>  [**Original Gaussian Grouping Project**](https://github.com/lkeab/gaussian-grouping.git)
> 
> [**Original Boundary Loss Project**](https://github.com/LIVIAETS/boundary-loss.git)


To run the model
Clone the repository locally
```
git clone https://github.com/Gijsvanmeer/GaussianGroupingOpenClosedVocabs.git
cd GaussianGroupingOpenClosedVocabs
```

The default, provided install method is based on Conda package and environment management, as per the original Gaussian Grouping project:
```bash
conda create -n gaussian_open_closed python=3.8 -y
conda activate gaussian_open_closed 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
