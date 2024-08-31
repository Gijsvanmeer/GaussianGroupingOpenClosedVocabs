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

conda install pytorch==1.12.0 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips
pip install requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```



Several scripts are provided to run the different parts of the code.
The main ones added are:
```
train_bloss.sh
train_combi.sh
train_contrast.sh
train_double_mlp.sh
render_closed_vocab.sh
obtain_MIOU.sh
```
All other scripts belong to the original Gaussian Grouping implementation and thus behave the same. The ```train_default.sh``` performs exactly the same as the ```train.sh``` in the original implementation.

Several parameters need to be set depending on which script is ran.
They are:
```
$0 output_name: Directory containing the model                      (str "Output_model" such that output/Output_Model is the target directory)
$1 dataset_name: Directory containing the main dataset              (str "Data_Set" such that data/Data_Set is the target directory)
$2 scale: Image scaling                                             (float 0-1)
$3 output_name: Directory name for model to be saved to             (str "Dir_Name" will create output/Dir_Name)
$4 closed_vocab_path: Directory in dataset containing the CV masks  (str)
$5 save_file: Location to save numerical values to                  (str location for numerical values to be saved, dont finish str with ".txt")
$6 lamda: Open vocabulary lambda value used for SV-MLP balance      (float 0-1)
$7 use_dl3 or use_closed_vocab: Whether to use SV-MLP               (Bool)
$8 method: Combi method secondary loss                              (str "Sobel" or "Canny")
$9 switch_iter: Switch iteration for contrastive method             (int 0-30000)
$10 filename: MIoU filename to save values to                       (str location for values to be saved, do include ".txt")
```
This finally gives the following ways of using the methods explained above:
```
train_bloss.sh $1 $2 $3 $4 $5 $6
train_combi.sh $1 $2 $3 $4 $5 $6 $7 $8
train_contrast.sh $1 $2 $3 $4 $5 $9
train_double_mlp.sh $1 $2 $3 $4 $5 $6
render_closed_vocab.sh $1 $4
obtain_MIOU.sh $3 $10 $7 $4
```

