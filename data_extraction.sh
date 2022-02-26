#!/bin/sh
curl -L "https://www.dropbox.com/s/3cpcvasgrs755wv/data.tar.gz?dl=1" > data.tar.gz
tar -xvzf data.tar.gz
rm -rf data.tar.gz
rm -rf data/data.tar.gz