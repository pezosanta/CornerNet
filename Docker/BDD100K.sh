#!/bin/bash

# Installing Google Drive downloading API with pip3
echo "Installing gdown"
pip3 install gdown

# Downloading Train, Val, Test datasets
echo "Downloading BDD100K dataset"

mkdir BDD100K
cd BDD100K

gdown https://drive.google.com/uc?id=1MymWKQUFCENauQP8A6QRk1EglinAKaud     # train
gdown https://drive.google.com/uc?id=1zgutvylvwv4CFz7rzlPFsL5mrTFHuqFG     # val
gdown https://drive.google.com/uc?id=1JS2jhgdEH8OwCOCQl-BqrYFuI0L2noaO     # test

echo "Unzipping Train, Val, Test datasets"
unzip -qq train.zip
unzip -qq val.zip
unzip -qq test.zip

rm -r train.zip
rm -r val.zip
rm -r test.zip

echo "Downloading Train, Val annotations"
gdown https://drive.google.com/uc?id=1JLkStcXlhVzvB7Fns-c2Wy_94j8NH75R     # train
gdown https://drive.google.com/uc?id=1fj9Sg4v4TwSvD2nxs90uzNqZgVythxLS     # val

cd ..

# Downloading CornerNet-Hourglass model parameters
mkdir ModelParams
cd ModelParams

echo "Downloading CornerNet-Hourglass model parameters"

mkdir hourglass
cd hourglass

gdown https://drive.google.com/uc?id=1HBFgPExTTlNTsAjyPoj9tMU7oqKaiuBz

cd ..

# Downloading MobileNetV3 model parameters
echo "Downloading MobileNetV3 model parameters"

mkdir mobilenetv3
cd mobilenetv3

gdown https://drive.google.com/uc?id=14mGBjXldBPoJfOndSxY_26Q4vC5HjVvk

cd ../..




