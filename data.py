# coding:utf-8
from numpy import *


# random embedding for words
def gen_rand_we(d):
    r = sqrt(6) / sqrt(51)
    temp = random.rand(d, 1) * 2 * r - r
    we = []
    for i in range(len(temp)):
        we.append(temp[i, 0])
    return array(we)


def gen_rand_we_(d):
    return random.uniform(0.0, 1.0, d)


# word2vec
def loadWord2Vec(path, delm="\t"):
    map = {}
    fp = open(path, "r", encoding='utf-8')
    for line in fp.readlines():
        ls = line.strip().split(delm)
        key = ""
        value = []
        if len(ls) < 10:
            continue
        for i in range(len(ls)):
            if i == 0:
                key = ls[0].strip()
            else:
                value.append(float(ls[i]))
        tmp = array(value)
        map[key] = tmp
    print('loadWord2Vec', len(map))
    return map


def gen_we(term, nb_dim, vobdic, word2vector):
    w2v = ""
    if nb_dim == 0:
        return []
    if term in word2vector:
        w2v = word2vector[term]
    else:
        w2v = gen_rand_we(nb_dim)
    if term in vobdic:
        w2v = vobdic[term]
    else:
        vobdic[term] = w2v
    return w2v


def init_sent_embedding_(wordlists, nb_width, vobdic, nb_dim, nb_pos, nb_dep, nb_par, word2vector, head, tail):
    nb_len = len(wordlists)
    arr = []
    if nb_len < nb_width:
        subtract = nb_width - nb_len
        div = int(subtract / 2)
        for i in range(nb_width):
            if i < div:
                arr.append(head)
            elif div <= i and i < div + nb_len:
                j = i - div
                w2v = []
                ws = wordlists[j].split("/")
                if len(ws) != 4:
                    w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
                    arr.append(array(w2v))
                    continue
                w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
                if nb_pos > 0:
                    w2v.extend(gen_we(ws[1] + "___", nb_pos, vobdic, {}))
                if nb_dep > 0:
                    w2v.extend(gen_we(ws[2] + "____", nb_dep, vobdic, {}))
                if nb_par > 0:
                    w2v.extend(gen_we(ws[3], nb_par, vobdic, word2vector))
                arr.append(array(w2v))
            else:
                arr.append(tail)
    else:
        for i in range(nb_width):
            w2v = []
            ws = wordlists[i].split("/")
            if len(ws) != 4:
                w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
                arr.append(array(w2v))
                continue
            w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
            if nb_pos > 0:
                w2v.extend(gen_we(ws[1] + "___", nb_pos, vobdic, {}))
            if nb_dep > 0:
                w2v.extend(gen_we(ws[2] + "____", nb_dep, vobdic, {}))
            if nb_par > 0:
                w2v.extend(gen_we(ws[3], nb_par, vobdic, word2vector))
            arr.append(array(w2v))
    return arr


def init_sent_embedding(wordlists, nb_width, vobdic, nb_dim, nb_pos, nb_dep, nb_par, word2vector, head):
    nb_len = len(wordlists)
    arr = []
    if nb_len < nb_width:
        subtract = nb_width - nb_len
        for i in range(nb_width):
            if i < subtract:
                arr.append(head)
            else:
                j = i - subtract
                w2v = []
                ws = wordlists[j].split("/")
                if len(ws) != 4:
                    w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
                    arr.append(array(w2v))
                    continue
                w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
                if nb_pos > 0:
                    w2v.extend(gen_we(ws[1] + "___", nb_pos, vobdic, {}))
                if nb_dep > 0:
                    w2v.extend(gen_we(ws[2] + "____", nb_dep, vobdic, {}))
                if nb_par > 0:
                    w2v.extend(gen_we(ws[3], nb_par, vobdic, word2vector))
                arr.append(array(w2v))
    else:
        for i in range(nb_width):
            w2v = []
            ws = wordlists[i].split("/")
            if len(ws) != 4:
                w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
                arr.append(array(w2v))
                continue
            w2v.extend(gen_we(ws[0], nb_dim, vobdic, word2vector))
            if nb_pos > 0:
                w2v.extend(gen_we(ws[1] + "___", nb_pos, vobdic, {}))
            if nb_dep > 0:
                w2v.extend(gen_we(ws[2] + "____", nb_dep, vobdic, {}))
            if nb_par > 0:
                w2v.extend(gen_we(ws[3], nb_par, vobdic, word2vector))
            arr.append(array(w2v))
    return arr


# 每一行都是一句分词后的问题和对应的标签
def load_sent_data(path, w2vPath, nb__sent, nb__width, nb__dim, nb__pos, nb__dep, nb__par, delm="\t"):
    fp = open(path, "r", encoding='utf-8')
    num = nb__sent
    nb_width = nb__width
    nb_dim = nb__dim
    nb_pos = nb__pos
    nb_dep = nb__dep
    nb_par = nb__par
    label = empty((num,), dtype="uint8")
    # data = np.empty((num,1,nb_width,nb_dim+nb_pos+nb_dep+nb_par),dtype="float32")
    data = empty((num, nb_width, nb_dim + nb_pos + nb_dep + nb_par), dtype="float32")
    index = 0
    labeldic = {}
    vobdic = {}
    labelindex = 0
    word2vector = loadWord2Vec(w2vPath, " ")
    head = gen_rand_we(nb_dim + nb_pos + nb_dep + nb_par)
    tail = gen_rand_we(nb_dim + nb_pos + nb_dep + nb_par)
    for line in fp:
        line = line.strip()
        if len(line) > 3:
            if (ord(line[0]) == 239 and ord(line[1]) == 187 and ord(line[2]) == 191):
                line = line[3:len(line)]
        ls = line.split("\t")
        if len(ls) != 2 or len(line) == 0:
            continue
        ws = ls[0].strip().split(" ")
        arr = []
        arr = init_sent_embedding(ws, nb_width, vobdic, nb_dim, nb_pos, nb_dep, nb_par, word2vector, head)
        assa = asarray(arr, dtype="float32")
        if not ls[1] in labeldic:
            labeldic[ls[1]] = labelindex
            labelindex = labelindex + 1
        label[index] = labeldic[ls[1]]
        # data[index,:,:,:]=assa
        data[index, :, :] = assa
        index = index + 1
    word2vector = {}
    print('labledic', len(labeldic))
    return data, label

# data ,label =load_data("C:\\Users\\jxhu\\PycharmProjects\\untitled\\CNN_sentence-master\\data\\mnist\\train\\")
# data2,label2 =load_sent_data("F:\\svn\\bmax5.0\\CNN\\data\\liantong.txt")
# print "ok"
