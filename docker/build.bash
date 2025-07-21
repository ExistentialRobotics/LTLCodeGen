#! /usr/bin/bash

docker build --rm -t ltlcodegen . --build-arg BASE_IMAGE=ubuntu:20.04 $@
