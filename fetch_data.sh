
https://download.pytorch.org/tutorial/hymenoptera_data.zip

#!/usr/bin/env bash

CODE_DIR=code
CNN_DIR=cnn
RNN_DIR=rnn

# Download images
IMAGES_DIR=$CODE_DIR/$CNN_DIR
mkdir $IMAGES_DIR
url=https://download.pytorch.org/tutorial/hymenoptera_data.zip
wget $url -O $IMAGES_DIR/hymenoptera_data.zip
unzip $IMAGES_DIR/hymenoptera_data.zip -d $IMAGES_DIR