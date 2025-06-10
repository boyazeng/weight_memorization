#!/bin/bash
clear
echo Downloading Hyper-Representations from https://zenodo.org/record/7529960/files/hyper_reps.zip?download=1

curl -# -L -o "hyper_reps.zip" "https://zenodo.org/record/7529960/files/hyper_reps.zip?download=1"

echo Unzipping hyper_reps

unzip hyper_reps.zip

echo Downloading pre-packaged zoos from https://zenodo.org/record/7529960/files/zoos.zip?download=1

curl -# -L -o "zoos.zip" "https://zenodo.org/record/7529960/files/zoos.zip?download=1"

echo Unzipping zoos

unzip zoos.zip

echo done
