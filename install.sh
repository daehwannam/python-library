#!/usr/bin/sh

THIS_FILE_PATH=`realpath $0`
THIS_DIR_PATH=`dirname $THIS_FILE_PATH`

SOURCE_PACKAGE_NAME="dhnamlib"
SOURCE_PACKAGE_PATH="$THIS_DIR_PATH/$SOURCE_PACKAGE_NAME"

TARGET_DIR_PATH=$1
TARGET_PACKAGE_NAME=$PACKAGE_NAME
TARGET_PACKAGE_PATH="$TARGET_DIR_PATH/$TARGET_PACKAGE_NAME"

ln -r -s "$SOURCE_PACKAGE_PATH" "$TARGET_PACKAGE_PATH"
# echo "$TARGET_PACKAGE_PATH" "$SOURCE_PACKAGE_PATH"
