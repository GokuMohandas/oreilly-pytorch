## Document Classification with Convolutional Neural Networks

### Start Crayon (TensorBoard Visualization: https://github.com/torrvision/crayon)
```bash
cd server
docker build -t crayon:latest -f Dockerfile .
docker run -d -p 8888:8888 -p 8889:8889 --name crayon crayon
Go to locahost:8888 for Tensorboard.
```


### Training
```bash
python main.py --mode=train --data_dir=data/ag_news_csv --learning_rate=1e-3 --decay_rate=0.99 --dropout_p=0.5
```

### Inference
```bash
python main.py --mode=infer --data_dir=data/ag_news_csv --model_name=model-ag_news_csv_1.00E-03_0.990000_0.50.pt
```