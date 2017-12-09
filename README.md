# [WIP]条件付確率場とベイズ階層言語モデルの統合による半教師あり形態素解析

修正版NPYCRFのC++実装です。

### Todo

- [x] NPYCRFの学習
- [ ] CRF単体の学習
- [ ] CRF単体の分割
- [ ] CRFの学習の並列化
- [ ] モデルの保存
- [ ] 評価
- [ ] 最大単語帳の予測に基づく枝刈り

# 動作環境

- Boost 1.65
- Python 3
- C++14
- python3-config

# 準備

## macOS

macOSの場合、PythonとBoostはともにbrewでインストールする必要があります。

### Python 3のインストール

```bash
brew install python3
```

`PYTHONPATH`を変更する必要があるかもしれません。

### Boostのインストール

```bash
brew install boost-python --with-python3
```

## Ubuntu

### Boostのインストール

```bash
./bootstrap.sh --with-python=python3 --with-python-version=3.5
./b2 python=3.5 -d2 -j4 --prefix YOUR_BOOST_DIR install
```

Pythonのバージョンを自身のものと置き換えてください。

## ビルド

以下のコマンドで`npycrf.so`が`/run/`に生成されます。

```bash
make install
```

`makefile`内のBoostのパスを環境に合わせて変更してください。

# 学習

読み込んだテキストから教師付きデータをランダムに選択する場合は`/run/split_data/train.py`を使います。

```
python3 train.py -file dataset.txt  -ssl-split 0.1 -td-split 0.9 -neologd /usr/local/lib/mecab/dic/mecab-ipadic-neologd
```

教師付きデータと教師なしデータをあらかじめ用意しておく場合は`/run/separate_data/train.py`を使います。

```
python3 train.py -file-l supervised.txt -file-u unsupervised.txt -td-split 0.9 -neologd /usr/local/lib/mecab/dic/mecab-ipadic-neologd
```

## 

## 注意事項

研究以外の用途には使用できません。

https://twitter.com/daiti_m/status/851810748263157760

実装に誤りが含まれる可能性があります。

質問等、何かありましたらissueにてお知らせください。

# 参考文献
- [条件付確率場とベイズ階層言語モデルの統合による半教師あり形態素解析](http://chasen.org/~daiti-m/paper/nlp2011semiseg.pdf)
- [半教師あり形態素解析 NPYCRF の修正](http://www.anlp.jp/proceedings/annual_meeting/2016/pdf_dir/D6-3.pdf)