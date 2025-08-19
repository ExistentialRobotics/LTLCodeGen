#! /usr/bin/bash

docker build --rm -t ltlcodegen . --build-arg BASE_IMAGE=nvidia/cuda:12.9.1-runtime-ubuntu20.04 $@
