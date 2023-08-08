from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from gensim.models import word2vec

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载模型
model = word2vec.Word2Vec.load('ckpt/word2vec.model')

word = "丹药"
topn = 50
keys = model.wv.similar_by_word(word, topn=topn)

# 绘制图片
fig = plt.figure()
ax = Axes3D(fig)
# 绘出关键词
x = model.wv[word][0]
y = model.wv[word][1]
z = model.wv[word][2]
ax.scatter(x, y, z)
ax.text(x, y, z, word)

# 绘出相似的词
for key in keys:
    x = model.wv[key[0]][0]
    y = model.wv[key[0]][1]
    z = model.wv[key[0]][2]
    ax.scatter(x, y, z)
    # ax.annotate(key[0], xy=(x, y, z), xytext=(x + 0.001, y + 0.001, z + 0.001))
    ax.text(x, y, z, key[0])

plt.show()
