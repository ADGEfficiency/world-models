 #!/usr/bin/env bash
pip install gdown -q
WMHOME="$HOME/world-models-experiments"
mkdir -p $WMHOME
FILEID="1qPjWfEbt_6-V2v7EsnFWHbTGomtV_MlY"
FILENAME="$WMHOME/pretrained-model.zip"
gdown --id $FILEID -O $FILENAME
unzip $FILENAME -d $WMHOME
rm -rf $WMHOME/__MACOSX
