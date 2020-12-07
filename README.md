# SRGANs: Using SRGANs to Generate Photo-Realistic Images
Keras implementation of "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
___

### 1. Set up the environment

Clone and get into the project folder:
```
git clone https://github.com/houseofai/srgan.git
cd srgan
```
Then, create an environment using conda with Python 3.9
```
conda create -n srgan python=3.9
```

Install the dependencies:
```
pip install -r requirements.txt
```
*Note that I use Tensorflow 2.5 from a nightly build with cuda 11.1. Feel free to switch Tensorflow version*

### 2. Dataset

### 2.1 Celeb-A

Download the Celeb-A dataset from:

`https://ml-ds.s3.eu-west-3.amazonaws.com/img_align_celeba.zip`

And unzip it into a `data` directory:
```
mkdir data
cd data
aria2c -x 16 https://ml-ds.s3.eu-west-3.amazonaws.com/img_align_celeba.zip
unzip img_align_celeba.zip
```

So, the path to the data folder is: `data/img_align_celeba`

### 3. Configure

To configure for training, update the configuration file in `config/original.yml`
While most parameters can remain as is, update:
- the batch size to fit to your GPU (I use 32 for a 24GB GPU memory)
- the data paths and name. use only absolut path


### 4. Train

Simply run:
```
python3 run.py
```

Checkpoints will be store in the checkpoints' folder
Intermediate generated images will be stored in images folder