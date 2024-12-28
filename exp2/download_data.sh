#!/bin/bash

FOLDER1="./tune_context/data"
FOLDER2="./tune_dialogue/data"

if [ ! -d "$FOLDER1" ]; then
    echo "Folder '$FOLDER1' does not exist. Creating it."
    mkdir -p "$FOLDER"
else
    echo "Folder '$FOLDER1' already exists."
fi

if [ ! -d "$FOLDER2" ]; then
    echo "Folder '$FOLDER2' does not exist. Creating it."
    mkdir -p "$FOLDER"
else
    echo "Folder '$FOLDER2' already exists."
fi

FILE1_URL="https://drive.google.com/uc?id=1y_iHWJqUKKqXYDfDJMhFgu0vRRk2Hc5p"
FILE2_URL="https://drive.google.com/uc?id=1FIVsfphH1O7tFwa-HnkPxOnqHoTzshDR"

FILE1_PATH="$FOLDER1/"
FILE2_PATH="$FOLDER2/"

echo "Downloading context data..."
gdown "$FILE1_URL" -O "$FILE1_PATH"
echo "Downloading dilaogue data..."
gdown "$FILE2_URL" -O "$FILE2_PATH"


