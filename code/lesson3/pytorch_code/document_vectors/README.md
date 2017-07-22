## Document Vectors

Note: Overfitting because we are using a very small dataset. Lower the learning rate
and increase the dataset to do proper training.

### Start Crayon (TensorBoard Visualization: https://github.com/torrvision/crayon)
```bash
cd server
docker build -t crayon:latest -f Dockerfile .
docker run -d -p 8888:8888 -p 8889:8889 --name crayon crayon
Go to locahost:8888 for Tensorboard.
```

### Training
```bash
python main.py --mode=train --data_dir=data/articles
```

### Inference
```bash
python main.py --mode=infer --data_dir=data/articles
```