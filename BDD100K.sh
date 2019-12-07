#!/bin/bash

echo "Creating BDD100K dataset"

mkdir BDD100K
mkdir ModelParams

cd BDD100K

# Tanító adathalmaz letöltése
export trainimagesid=1MymWKQUFCENauQP8A6QRk1EglinAKaud
export trainimagesfilename=train.zip
wget --save-cookies trainimagescookies.txt 'https://docs.google.com/uc?export=download&id='$trainimagesid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > trainimagesconfirm.txt
wget --load-cookies trainimagescookies.txt -O $trainimagesfilename \
     'https://docs.google.com/uc?export=download&id='$trainimagesid'&confirm='$(<trainimagesconfirm.txt)


# Validációs adathalmaz letöltése
export valimagesid=1zgutvylvwv4CFz7rzlPFsL5mrTFHuqFG
export valimagesfilename=val.zip
wget --save-cookies valimagescookies.txt 'https://docs.google.com/uc?export=download&id='$valimagesid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > valimagesconfirm.txt
wget --load-cookies valimagescookies.txt -O $valimagesfilename \
     'https://docs.google.com/uc?export=download&id='$valimagesid'&confirm='$(<valimagesconfirm.txt)


# Tanító annotáció letöltése
export trainannotationid=1JLkStcXlhVzvB7Fns-c2Wy_94j8NH75R
export trainannotationfilename=bdd100k_labels_images_train.json
wget --save-cookies trainannotationcookies.txt 'https://docs.google.com/uc?export=download&id='$trainannotationid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > trainannotationconfirm.txt
wget --load-cookies trainannotationcookies.txt -O $trainannotationfilename \
     'https://docs.google.com/uc?export=download&id='$trainannotationid'&confirm='$(<trainannotationconfirm.txt)


# Validációs annotáció letöltése
export valannotationid=1fj9Sg4v4TwSvD2nxs90uzNqZgVythxLS
export valannotationfilename=bdd100k_labels_images_val.json
wget --save-cookies valannotationcookies.txt 'https://docs.google.com/uc?export=download&id='$valannotationid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > valannotationconfirm.txt
wget --load-cookies valannotationcookies.txt -O $valannotationfilename \
     'https://docs.google.com/uc?export=download&id='$valannotationid'&confirm='$(<valannotationconfirm.txt)

unzip train.zip
unzip val.zip

cd ..
cd ModelParams

# Best ModelParams letöltése
export modelparamsid=13tYTwt-1PL8e-tCBN8QBC2vCNmEgbkmu
export modelparamsfilename=train_valid_pretrained_cornernet-epoch3-iter5067.pth
wget --save-cookies modelparamscookies.txt 'https://docs.google.com/uc?export=download&id='$modelparamsid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > modelparamsconfirm.txt
wget --load-cookies modelparamscookies.txt -O $modelparamsfilename \
     'https://docs.google.com/uc?export=download&id='$modelparamsid'&confirm='$(<modelparamsconfirm.txt)

cd ..
