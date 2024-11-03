#!/bin/bash

cd bin
cmake .
make -j
./main
cd ..