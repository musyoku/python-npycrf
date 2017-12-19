# [WIP]条件付確率場とベイズ階層言語モデルの統合による半教師あり形態素解析

修正版NPYCRFのC++実装です。

L-BFGSの実装がまだできていないので代わりにSGDでCRFを最適化します。

NPYLMは3-gramで固定です。

- [条件付確率場とベイズ階層言語モデルの統合による半教師あり形態素解析](http://chasen.org/~daiti-m/paper/nlp2011semiseg.pdf)
- [半教師あり形態素解析 NPYCRF の修正](http://www.anlp.jp/proceedings/annual_meeting/2016/pdf_dir/D6-3.pdf)
- [実装について](http://musyoku.github.io/2017/12/19/npycrf/)

#### Todo

- [x] NPYCRFの学習
- [x] モデルの保存
- [x] 評価
- [x] NPYLM単体による単語分割
- [ ] L-BFGS
- [ ] CRF単体の学習
- [ ] CRF単体による単語分割
- [ ] CRFの学習の並列化
- [ ] 最大単語帳の予測に基づく枝刈り

# 動作環境

- Boost 1.65
- Python 3
- C++14
- python3-config

# 準備

## macOS

macOSの場合、PythonとBoostはともにbrewでインストールする必要があります。

#### Python 3のインストール

```bash
brew install python3
```

`PYTHONPATH`を変更する必要があるかもしれません。

#### Boostのインストール

```bash
brew install boost-python --with-python3
```

## Ubuntu

#### Boostのインストール

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

読み込んだテキストから教師付きデータをランダムに選択する場合は`/run/split_file/train.py`を使います。

```
python3 train.py -file dataset.txt -ssl-split 0.1 -td-split 0.9 -neologd /usr/local/lib/mecab/dic/mecab-ipadic-neologd
```

教師付きデータと教師なしデータをあらかじめ用意しておく場合は`/run/separate_files/train.py`を使います。

```
python3 train.py -file-l supervised.txt -file-u unsupervised.txt -td-split 0.9 -neologd /usr/local/lib/mecab/dic/mecab-ipadic-neologd
```

学習の一時停止や再開はできません。

# 分割

ビタビアルゴリズムによる最尤分割を求めます。

```
python3 viterbi.py -file test.txt -neologd /usr/local/lib/mecab/dic/mecab-ipadic-neologd
```

# 注意事項

研究以外の用途には使用できません。

https://twitter.com/daiti_m/status/851810748263157760

実装に誤りが含まれる可能性があります。

質問等、何かありましたらissueにてお知らせください。

# 結果

[日本語版text8コーパス](http://hironsan.hatenablog.com/entry/japanese-text8-corpus)から1万文を教師付きデータとして用い、2chの[けものフレンズ](https://shiba.5ch.net/test/read.cgi/anime/1490363261)などのスレッドから7万文を教師なしデータとして用いました。

以下は教師なしデータに対するMeCabと学習後のNPYCRFの分割結果の一部です。

```
MeCab+NEologd
サーバル / を / 救出 / し / た / 瞬間 / に / 黒 / セルリアン / の / 前足 / が / 砕け / た / の / は / 、
NPYCRF
サーバル / を / 救出 / し / た / 瞬間 / に / 黒 / セルリアン / の / 前足 / が / 砕け / た / の / は / 、

MeCab+NEologd
ガイドブック / 発送 / さ / れ / て / た / けど / 初版 / は / 保存 / し / て / おき / たい / 症候群 / が / 発症 / し / て / しまい / そう / だ / わ
NPYCRF
ガイドブック / 発送 / さ / れ / て / た / けど / 初版 / は / 保存 / し / て / おき / たい / 症候群 / が / 発症 / し / て / しまい / そう / だ / わ

MeCab+NEologd
iq / が / 溶ける / アニメ / 、 / と / 揶揄 / さ / れ / た / の / は / それだけ / 動物 / の / 視点 / に / 寄り添え / た
NPYCRF
iq / が / 溶け / る / アニメ / 、 / と / 揶揄 / さ / れ / た / の / は / それ / だけ / 動物 / の / 視点 / に / 寄り / 添 / え / た

MeCab+NEologd
人間 / から / すれ / ば / 紙飛行機 / を / 作っ / たり / 、 / 遊び / を / 考える / の / は / 簡単 / な / こと / だ / けど
NPYCRF
人間 / から / すれ / ば / 紙 / 飛行機 / を / 作っ / た / り / 、 / 遊び / を / 考える / の / は / 簡単 / な / こと / だ / けど

MeCab+NEologd
こんな / こと / 今 / の / アニメ業界 / で / は / でき / ん / の / だろ / う / なっ / て
NPYCRF
こん / な / こと / 今 / の / アニメ業界 / で / は / でき / ん / の / だろ / う / な / っ / て

MeCab+NEologd
今更 / 気付い / た / ん / だ / けど / 野生 / 解放 / って / ヘラジカ / と / 戦う / 時 / ライオン / も / し / て / た / よ / ね / ?
NPYCRF
今更 / 気付 / い / た / んだけど / 野生 / 解放 / っ / て / ヘラジ / カ / と / 戦う / 時 / ライオン / も / し / て / た / よ / ね / ?

MeCab+NEologd
今日 / やっ / てる / # / 話 / タイムシフト / 予約 / 出来る / の / ?
NPYCRF
今日 / やっ / て / る / # / 話 / タイムシフト / 予約 / 出来る / の / ?

MeCab+NEologd
シリアス / や / バトル / アクション / だって / 日頃 / 好ん / でる / わけ / じゃ / ない
NPYCRF
シリアス / や / バトル / アクション / だ / っ / て / 日頃 / 好ん / で / る / わけ / じゃ / ない

MeCab+NEologd
最後 / アライ / さん / に / 活躍 / し / て / もらい / たい
NPYCRF
最後 / アライさん / に / 活躍 / し / て / もら / い / たい

MeCab+NEologd
フェネック / は / # / 番 / 目 / くらい / に / 頭 / が / いい / らしい
NPYCRF
フェネック / は / # / 番 / 目 / くらい / に / 頭 / が / いい / らしい

MeCab+NEologd
ニコニコ / の / お陰 / か / この / スレ / の / スタートダッシュ / は / なかなか / だ / ね
NPYCRF
ニコニコ / の / お陰 / か / この / スレ / の / スタート / ダッシュ / は / なかなか / だ / ね

MeCab+NEologd
デジタル / 版 / が / あれ / ば / 一番 / いい / ん / だ / けど / ねぇ
NPYCRF
デジタル / 版 / が / あれ / ば / 一番 / いい / んだけど / ね / ぇ

MeCab+NEologd
ちょっと / 高い / 公式ガイドブック / だ / から / けものフレンズ / の / 人気 / なら / # / 万 / は / 売れる / と / 思う / よ
NPYCRF
ちょっ / と / 高い / 公式 / ガイドブック / だ / から / けものフレンズ / の / 人気 / なら / # / 万 / は / 売 / れる / と / 思う / よ

MeCab+NEologd
パーク / 中 / の / ボス / が / 集結 / 、 / 変形 / 、 / 合体 / し / て / 巨大 / な / 真 / ボス / に / なっ / て / 黒セ / ルリアン / を / やっつける / 。
NPYCRF
パーク / 中 / の / ボス / が / 集結 / 、 / 変形 / 、 / 合体 / し / て / 巨大 / な / 真 / ボス / に / なっ / て / 黒 / セルリアン / を / やっつけ / る / 。
```