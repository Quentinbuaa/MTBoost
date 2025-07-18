#!/bin/bash

cd /root/exp ||exit 1

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

git add *
git diff --quiet || git commit -m "Auto update: $TIMESTAMP"
git push -u origin master
