#!/bin/bash

# Installing Google Drive downloading API with pip3
echo "Installing gdown"
pip3 install gdown

# Downloading Train, Val, Test datasets
echo "Downloading BDD100K dataset"

mkdir BDD100K
cd BDD100K

gdown https://drive.google.com/uc?id=1qNQlsitLh04VYiNIF-E-nxfGXbzBxfPR     # train
gdown https://drive.google.com/uc?id=1LGtbO9pqHNFjSFFBlfQjdLNkfFVyjjg7     # val
gdown https://drive.google.com/uc?id=1o0zHZn6QCthqAoLo1DEIxd1yHKt4oQQp     # test

echo "Unzipping Train, Val, Test datasets"
unzip -qq train.zip
unzip -qq val.zip
unzip -qq test.zip

rm -r train.zip
rm -r val.zip
rm -r test.zip

echo "Downloading Train, Val annotations"
gdown https://drive.google.com/uc?id=1qbPP3QdSSXHbUI5YoozQlTURk_kNXDpN     # train
gdown https://drive.google.com/uc?id=1R4mzlc0u9CmDA-W92DlXtKhlacypiW8I     # val

cd ..

# Downloading CornerNet-Hourglass model parameters
mkdir ModelParams
cd ModelParams

echo "Downloading CornerNet-Hourglass model parameters"

mkdir hourglass
cd hourglass

gdown https://drive.google.com/uc?id=1dWqYHrGsYMWVY0w_NZFXCVzfORb-i4sU

cd ..

# Downloading MobileNetV3 model parameters
echo "Downloading MobileNetV3 model parameters"

mkdir mobilenetv3
cd mobilenetv3

gdown https://drive.google.com/uc?id=1i7fxnyWRiuRqQ6MeR0LzBNdi0MzgVCF-

cd ../..




