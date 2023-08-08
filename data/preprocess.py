import jieba

with open('raw/凡人修仙传.txt', 'r', encoding="utf-8") as r, \
        open('./凡人修仙传_processed.txt', 'w', encoding="utf-8") as w:
    for line in r.read().split("\n\n"):
        if (r == "\n"):
            continue
        jieba_out = jieba.cut(line.strip(), cut_all=False, HMM=True)  # 精确模式
        w.write(" ".join(jieba_out) + "\n")

