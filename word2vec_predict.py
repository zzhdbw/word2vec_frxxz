from gensim.models import word2vec

model = word2vec.Word2Vec.load('ckpt/word2vec.model')

# 相似度计算
sim1 = model.wv.similarity('修仙', '炼丹')
print(sim1)
sim2 = model.wv.similarity('修仙', '西瓜')
print(sim2)

# 前10相似度的词
for key in model.wv.similar_by_word(u'修仙', topn=10):
    print(key)

# 获取词向量
print(model.wv["修仙"])
print(model.wv["修仙"].size)


# 画图-2d
# word = "道友"
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#
# keys = model.wv.similar_by_word(word, topn=20)
#
#
# for key in keys:
#     x = model.wv[key[0]][0]
#     y = model.wv[key[0]][1]
#     plt.scatter(x, y)
#     plt.annotate(key[0], xy=(x, y), xytext=(x + 0.001, y + 0.001))
#
#
# y = model.wv[word][1]
# x = model.wv[word][0]
# plt.scatter(x, y)
# plt.annotate(word, xy=(x, y), xytext=(x + 0.001, y + 0.001))
# plt.show()
