import argparse, sys, os, time, codecs, random, re
from tabulate import tabulate
import MeCab
import npycrf as nlp


def main():
    assert args.working_directory is not None
    try:
        os.mkdir(args.working_directory)
    except:
        pass

    # 辞書
    dictionary = nlp.dictionary(
        os.path.join(args.working_directory, "char.dict"))

    # モデル
    crf = nlp.crf(os.path.join(args.working_directory, "crf.model"))
    npylm = nlp.npylm(os.path.join(args.working_directory, "npylm.model"))
    npycrf = nlp.npycrf(npylm=npylm, crf=crf)

    num_features = crf.get_num_features()
    num_character_ids = dictionary.get_num_characters()
    print(
        tabulate([["#characters", num_character_ids],
                  ["#features", num_features]]))

    # ビタビアルゴリズムによる最尤分解を求める
    assert args.test_filename is not None or args.test_directory is not None
    sentence_list = []

    def preprocess(sentence):
        sentence = re.sub(r"[0-9.,]+", "#", sentence)
        sentence = sentence.strip()
        return sentence

    if args.test_filename is not None:
        with codecs.open(args.test_filename, "r", "utf-8") as f:
            for sentence_str in f:
                sentence_str = preprocess(sentence_str)
                sentence_list.append(sentence_str)

    if args.test_directory is not None:
        for filename in os.listdir(args.test_directory):
            with codecs.open(
                    os.path.join(args.test_directory, filename), "r",
                    "utf-8") as f:
                for sentence_str in f:
                    sentence_str = preprocess(sentence_str)
                    sentence_list.append(sentence_str)

    # 教師データはMeCabによる分割
    tagger = MeCab.Tagger() if args.neologd_path is None else MeCab.Tagger(
        "-d " + args.neologd_path)

    # バグ回避のため空データを分割
    tagger.parse("")

    for sentence_str in sentence_list:
        sentence_str = sentence_str.strip()
        m = tagger.parseToNode(sentence_str)  # 形態素解析
        words_true = []
        while m:
            word = m.surface
            if len(word) > 0:
                words_true.append(word)
            m = m.next
        if len(words_true) > 0:
            words_npycrf = npycrf.parse(sentence_str, dictionary)
            words_npylm = npylm.parse(sentence_str, dictionary)
            print("MeCab+NEologd")
            print(" / ".join(words_true))
            print("NPYCRF")
            print(" / ".join(words_npycrf))
            print("NPYLM")
            print(" / ".join(words_npylm))
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 以下のどちらかを必ず指定
    parser.add_argument("--test-filename", "-file", type=str, default=None)
    parser.add_argument("--test-directory", "-dir", type=str, default=None)

    parser.add_argument(
        "--working-directory",
        "-cwd",
        type=str,
        default="out",
        help="ワーキングディレクトリ")
    parser.add_argument("--neologd-path", "-neologd", type=str, default=None)
    args = parser.parse_args()
    main()
