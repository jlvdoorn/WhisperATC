#!/bin/bash

echo "Installing dependencies"
wget https://www.7-zip.org/a/7z2301-linux-x64.tar.xz
mkdir 7z
tar -xf 7z2301-linux-x64.tar.xz -C ./7z
rm -rf 7z2301-linux-x64.tar.xz

echo "Downloading ATCO2-ASR"
wget https://www.replaywell.com/atco2/download/ATCO2-ASRdataset-v1_beta.tgz
mkdir ATCO2-ASR
tar -xzvf ATCO2-ASRdataset-v1_beta.tgz
mv ATCO2-ASRdataset-v1_beta/* ATCO2-ASR
rm -rf ATCO2-ASRdataset-v1_beta.tgz
rm -rf ATCO2-ASRdataset-v1_beta

echo "Downloading ATCOSIM"
wget http://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/atcosim.iso
mkdir ATCOSIM
./7z/7zz x atcosim.iso -oATCOSIM
rm -rf atcosim.iso

echo "Cleaning up"
rm -rf ./7z