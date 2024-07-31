#!/bin/bash

wget https://www.sqlite.org/2021/sqlite-autoconf-3350500.tar.gz
tar -xzf sqlite-autoconf-3350500.tar.gz
cd sqlite-autoconf-3350500
./configure
make
make install
cd ..
rm -rf sqlite-autoconf-3350500 sqlite-autoconf-3350500.tar.gz