CC = g++
BOOST = /usr/local/Cellar/boost/1.65.0
INCLUDE = `python3-config --includes` -std=c++14 -I$(BOOST)/include
LDFLAGS = `python3-config --ldflags` -lboost_serialization -lboost_python3 -L$(BOOST)/lib
SOFLAGS = -shared -fPIC -march=native

install: ## npylm.soを生成
	$(CC) $(INCLUDE) $(SOFLAGS) src/python.cpp src/python/*.cpp src/python/model/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp $(LDFLAGS) -o run/npylm.so -O3
	cp run/npylm.so run/semi-supervised/npylm.so
	cp run/npylm.so run/unsupervised/npylm.so
	rm -rf run/npylm.so

install_ubuntu: ## npylm.soを生成
	$(CC) -Wl,--no-as-needed -Wno-deprecated $(INCLUDE) $(SOFLAGS) src/python.cpp src/python/*.cpp src/python/model/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp $(LDFLAGS) -o run/npylm.so -O3
	cp run/npylm.so run/semi-supervised/npylm.so
	cp run/npylm.so run/unsupervised/npylm.so
	rm -rf run/npylm.so

check_includes:	## Python.hの場所を確認
	python3-config --includes

check_ldflags:	## libpython3の場所を確認
	python3-config --ldflags

module_tests: ## 各モジュールのテスト.
	$(CC) test/module_tests/crf/crf.cpp src/npycrf/*.cpp src/npycrf/crf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/module_tests/crf/crf $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/crf/crf
	$(CC) test/module_tests/npylm/wordtype.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/module_tests/npylm/wordtype $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/wordtype
	$(CC) test/module_tests/npylm/npylm.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/module_tests/npylm/npylm $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/npylm
	$(CC) test/module_tests/npylm/vpylm.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/module_tests/npylm/vpylm $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/vpylm
	$(CC) test/module_tests/npylm/sentence.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/module_tests/npylm/sentence $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/sentence
	$(CC) test/module_tests/npylm/hash.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/module_tests/npylm/hash $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/npylm/hash

running_tests:	## 運用テスト
	$(CC) test/running_tests/likelihood.cpp src/python/*.cpp src/python/model/*.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/running_tests/likelihood $(INCLUDE) $(LDFLAGS) -O0 -g -Wall
	$(CC) test/running_tests/save.cpp src/python/*.cpp src/python/model/*.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp -o test/running_tests/save $(INCLUDE) $(LDFLAGS) -O0 -g -Wall
	$(CC) test/running_tests/train.cpp src/python/*.cpp src/python/model/*.cpp src/npycrf/*.cpp src/npycrf/npylm/*.cpp src/npycrf/npylm/lm/*.cpp src/npycrf/crf/*.cpp  -o test/running_tests/train $(INCLUDE) $(LDFLAGS) -O0 -g -Wall

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help