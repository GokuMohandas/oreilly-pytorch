# oreilly-pytorch
Introductory PyTorch Tutorials

### Setup Environment
```bash
virtualenv -p python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install PyTorch Components
```bash
pip install http://download.pytorch.org/whl/torch-0.1.11.post5-cp27-none-macosx_10_7_x86_64.whl
pip install torchvision
```

### Using Docker
```bash
docker build -t gokumd/oreilly-pytorch:gpu -f Dockerfile.gpu .
nvidia-docker run -it --ipc=host -p 8888:8888 -p 6006:6006 oreilly-pytorch:gpu bash
```