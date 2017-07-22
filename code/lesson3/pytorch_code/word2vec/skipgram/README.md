## Skip-gram Model

Note: This is a naive skip-gram implementation. It includes subsampling but does not use a sampling based loss. It is purely for a simple demonstration. If you are looking to train useful embeddings, you should either use pre-trained word2vec or GloVe embeddings, which is covered in Section 4.3

### Start Crayon (TensorBoard Visualization: https://github.com/torrvision/crayon)
```bash
cd server
docker build -t crayon:latest -f Dockerfile .
docker run -d -p 8888:8888 -p 8889:8889 --name crayon crayon
Go to locahost:8888 for Tensorboard.
```

### Data
```bash
Data from http://www.textfiles.com/etext/FICTION/wizrd_oz
```

### Training
```bash
python main.py --mode=train --data_dir=data/oz --data_file=oz.txt
```

### Inference
```bash
python main.py --mode=infer --data_dir=data/oz --data_file=oz.txt
```