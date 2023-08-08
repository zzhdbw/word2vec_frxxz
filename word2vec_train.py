from gensim.models import word2vec

sentences = word2vec.Text8Corpus("data/凡人修仙传_processed.txt")

# sentences  第一个参数是预处理后的训练语料库。是可迭代列表，但是对于较大的语料库，可以考虑直接从磁盘/网络传输句子的迭代。
# sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。
# size(int) 是输出词向量的维数，默认值是100。这个维度的取值与我们的语料的大小相关，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间，不过见的比较多的也有300维的。
# window（int）是一个句子中当前单词和预测单词之间的最大距离，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。windows越大所需要枚举的预测此越多，计算的时间越长。min_count 忽略所有频率低于此值的单词。默认值为5。
# workers表示训练词向量时使用的线程数,默认是当前运行机器的处理器核数。
# 还有关采样和学习率的，一般不常设置negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
# hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
# sg=2 代表skip-gram
model = word2vec.Word2Vec(sentences,
                          sg=1,
                          vector_size=100,
                          hs=1,
                          min_count=2,
                          window=5,
                          workers=6,
                          epochs=10
                          )

model.save(r'ckpt/word2vec.model')
