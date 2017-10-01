from numpy import *


def load_data_set():
    # 每个词是一个基本单元，每一行是一个句子（文档），整个list是文档（文档库）
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0 ,1 ,0 ,1 ,0 ,1  ]# 1:侮辱性文字 0：正常言论
    return posting_list, class_vec


def create_vocab_list(data_set):
    # 把数据集除去重复的词，返回一个列表：词汇表
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_word2vec(vocab_list, input_set):
    # 检查每一行句子中的每个单词是否出现，如果是，则数字化词汇表return_vec里相应的行的相应位置为1，其他不变，
    # 有几个单词这一行就有几个位置为1，最后得到这行句子的每个单词出现的频率列表
    # 如果遍历整个文档，就得到一个两维矩阵，每一行对应一个句子，每个列对应该位置的词是否出现，1出现，0没出现。
    return_vec = [0 ] *len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word :%s is not in my Vocabulary!" % word)
    return return_vec


def train_nb_0(train_matrix, train_category):
    """
    求概率
    :param train_matrix: 输入的已完成数字化的文档
    :param train_category: 人工标识的侮辱性句子的列表
    :return:
    """
    num_train_docs = len(train_matrix)  # 有几行代表文档有几个句子
    num_words = len(train_matrix[0])  # 每一行的长度代表词汇表一共有多少词32
    p_abusive = sum(train_category)/float(num_train_docs)
    p0_num = zeros(num_words)  # 建立一个32位的数组，记录出现0的频率
    p1_num = zeros(num_words)
    p0_denom = 0.0
    p1_denom = 0.0
    for i in range(num_train_docs):  # 遍历所有6个句子
        if train_category[i] == 1:  # 检查是否是侮辱性之句子，
            # 是则把文档相应的行放到p1_num,下次相加就是相应位置相加，最后求出p1
            # p1的每个位置表示该词出现的次数
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])  # 每个句子长度不一，相加表示p1所在
            # 句子一共有多少个单词
        else:
            p0_num += train_matrix[i]  # 检查是否是侮辱性之句子，
            # 不是则把文档相应的行放到p0_num,下次相加就是相应位置相加
            p0_denom += sum(train_matrix[i])

    p1_vec = p1_num/p1_denom
    p0_vec = p0_num/p0_denom  # 每个位置表示每个词出现次数除以非侮辱性文章的总词数=
    # 该词的概率，列表是所有词的概率列表

    return p0_vec, p1_vec,p_abusive



"""main"""


list_o_posts, list_classes = load_data_set()
my_vocab_list = create_vocab_list(list_o_posts)
print(my_vocab_list)
# train_mat = []
#
# for post_in_doc in list_o_posts:
#     train_mat.append(set_of_word2vec(my_vocab_list,post_in_doc))
# p0_v,p1_v,p_ab= train_nb_0(train_mat, list_classes)
# print("p_ab:")
# print( p_ab)
# print("po_v:")
# print( p0_v)
# print("p1_v:")
# print( p1_v)

# my_i = set_of_word2vec(my_vocab_list ,list_o_posts[1])
# print(my_i)
