SHELL := /bin/bash
CFLAGS=-Wall -Wno-overloaded-virtual -stdlib=libstdc++ -O2 `pkg-config --cflags opencv`
CXXFLAGS=$(CFLAGS)
LDFLAGS=`pkg-config --libs opencv`
