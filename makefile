CC = g++
BOOST = /usr/local/Cellar/boost/1.65.0
INCLUDE = `python3-config --includes` -std=c++14 -I$(BOOST)/include
LDFLAGS = `python3-config --ldflags` -lboost_serialization -lboost_python3 -L$(BOOST)/lib
SOFLAGS = -shared -fPIC -march=native
SOURCES = 	src/python/*.cpp \
			src/python/model/*.cpp \
			src/npycrf/*.cpp \
			src/npycrf/npylm/*.cpp \
			src/npycrf/npylm/lm/*.cpp \
			src/npycrf/crf/*.cpp \
			src/npycrf/solver/*.cpp

install: ## npycrf.soを生成
	$(CC) $(INCLUDE) $(SOFLAGS) src/python.cpp $(SOURCES) $(LDFLAGS) -o run/npycrf.so -O3
	cp run/npycrf.so run/split_data/npycrf.so
	cp run/npycrf.so run/separate_data/npycrf.so
	rm -rf run/npycrf.so

install_ubuntu: ## npycrf.soを生成
	$(CC) -Wl,--no-as-needed -Wno-deprecated $(INCLUDE) $(SOFLAGS) src/python.cpp $(SOURCES) $(LDFLAGS) -o run/npycrf.so -O3
	cp run/npycrf.so run/split_data/npycrf.so
	cp run/npycrf.so run/separate_data/npycrf.so
	rm -rf run/npycrf.so
	
check_includes:	## Python.hの場所を確認
	python3-config --includes

check_ldflags:	## libpython3の場所を確認
	python3-config --ldflags

module_tests: ## 各モジュールのテスト.
	$(CC) test/module_tests/solver/sgd.cpp $(SOURCES) -o test/module_tests/solver/sgd $(INCLUDE) $(LDFLAGS) -O3
	./test/module_tests/solver/sgd
	$(CC) test/module_tests/npylm/lattice.cpp $(SOURCES) -o test/module_tests/npylm/lattice $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/lattice
	$(CC) test/module_tests/npylm/vpylm.cpp $(SOURCES) -o test/module_tests/npylm/vpylm $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/vpylm
	$(CC) test/module_tests/crf/crf.cpp $(SOURCES) -o test/module_tests/crf/crf $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/crf/crf
	$(CC) test/module_tests/npylm/wordtype.cpp $(SOURCES) -o test/module_tests/npylm/wordtype $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/wordtype
	$(CC) test/module_tests/npylm/npylm.cpp $(SOURCES) -o test/module_tests/npylm/npylm $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/npylm
	$(CC) test/module_tests/npylm/sentence.cpp $(SOURCES) -o test/module_tests/npylm/sentence $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/sentence
	$(CC) test/module_tests/npylm/hash.cpp $(SOURCES) -o test/module_tests/npylm/hash $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/hash

running_tests:	## 運用テスト
	$(CC) test/running_tests/train.cpp $(SOURCES)  -o test/running_tests/train $(INCLUDE) $(LDFLAGS) -O3 -Wall
	$(CC) test/running_tests/likelihood.cpp $(SOURCES) -o test/running_tests/likelihood $(INCLUDE) $(LDFLAGS) -O0 -g -Wall
	$(CC) test/running_tests/save.cpp $(SOURCES) -o test/running_tests/save $(INCLUDE) $(LDFLAGS) -O0 -g -Wall

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help