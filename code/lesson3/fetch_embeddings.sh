# Make directory to store embeddings
EMBEDDINGS_DIR=embeddings
mkdir $EMBEDDINGS_DIR

# Download GloVe
GLOVE_DIR=$EMBEDDINGS_DIR/glove
mkdir $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR