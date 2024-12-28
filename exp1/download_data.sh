#!/bin/bash

FOLDER="./tune_dialogue/data"

if [ ! -d "$FOLDER" ]; then
    echo "Folder '$FOLDER' does not exist. Creating it."
    mkdir -p "$FOLDER"
else
    echo "Folder '$FOLDER' already exists."
fi


FILE_URL="https://drive.google.com/uc?id=15gCmyfk9RWHchUWEZbQ4DL-AdMVx2zvI"
FILE_PATH="$FOLDER/"

echo "Downloading context data..."
gdown "$FILE_URL" -O "$FILE_PATH"


