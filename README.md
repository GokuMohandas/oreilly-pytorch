# oreilly-pytorch
Introductory PyTorch Tutorials
- Alfredo Canziani – [@alfcnz](https://twitter.com/alfcnz)
- Goku Mohandas – [@GokuMohandas](https://twitter.com/gokumohandas)

### Environment Options
```bash
1. Docker container (recommended)
2. Local Machine
```

### Option 1: Docker
#### CPU
```bash
docker build -t gokumd/pytorch-docker:cpu -f Dockerfile.gpu .
docker run -it --name=nlpbook --ipc=host -p 8888:8888 -p 6006:6006 gokumd/pytorch-docker:cpu
```
#### GPU
```bash
docker build -t gokumd/pytorch-docker:gpu -f Dockerfile.gpu .
nvidia-docker run -it --ipc=host -p 8888:8888 -p 6006:6006 gokumd/pytorch-docker:gpu
```
#### Setup virtualenv:
```bash
virtualenv -p python3.6 venv
source venv/bin/activate
pip install numpy==1.12.1
pip install requests==2.13.0
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl
pip install torchvision
```

### Option 2: Local machine
#### OSX:
```bash
virtualenv -p python3.5 venv
source venv/bin/activate
pip install numpy==1.12.1
pip install requests==2.13.0
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/torch-0.1.12.post2-cp35-cp35m-macosx_10_7_x86_64.whl 
pip install torchvision
```
#### Linux:
```bash
virtualenv -p python3.6 venv
source venv/bin/activate
pip install numpy==1.12.1
pip install requests==2.13.0
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl
pip install torchvision
```

### Conda
If you are use the Anaconda python distribution, follow these instructions to setup the docker container and then the virtual environment.
```bash
clone https://github.com/pytorch/pytorch in the root of this repo and replace the Dockerfile with our Dockerfile.conda
docker build -t gokumd/pytorch-docker-conda:cpu -f Dockerfile.conda .
docker run -it --ipc=host -p 8888:8888 -p 6006:6006 gokumd/pytorch-docker-conda:cpu
conda update conda
conda create -n venv python=3.5 anaconda
source activate venv
conda install --yes --file requirements.txt
```

### Set Up Crayon BEFORE jupyter (Port: 8888) - https://github.com/torrvision/crayon
```bash
cd server
docker build -t crayon:latest -f Dockerfile .
docker run -d -p 8888:8888 -p 8889:8889 --name crayon crayon
Go to locahost:8888 for Tensorboard.
```

### Start IPython/Jupyter Notebook (Port: 8889)
```bash
jupyter notebook --allow-root
```

### Common Docker Issues
```bash
If ports are occupied:
    lsof -nP +c 15 | grep LISTEN
    sudo kill -9 <>
```

