#!/usr/bin/python
# -*- coding: UTF-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom
import codecs, json, os, re
import hyperparams as hp
from util import is_Chinese


def loadOneXml(xml_file, xml2json, reload):
    """
    parser xml file to json format and save in 'temp/'

    :param xml_file: the source xml file
    :param xml2json:save xml to json format
    :param reload: if reload is True, the save file will be constructed newly

    :return text: string, the parsered data
    :return PlatoNamedEntityUIMA: list, the parsered data
    :return PlatoRelationUIMA: list, the parsered data
    """
    # parser text, Sentence, BaseToken, PlatoNamedEntityUIMA, PlatoRelationUIMA from xml file
    Sentence, BaseToken, PlatoNamedEntityUIMA, PlatoRelationUIMA = [], [], [], []

    # parser
    DOMTree = xml.dom.minidom.parse(xml_file)
    xmi = DOMTree.documentElement

    # get text, Sentence, BaseToken, PlatoNamedEntityUIMA, PlatoRelationUIMA
    nodelist = xmi.getElementsByTagName("cas:Sofa")
    text = nodelist[0].getAttribute("sofaString")
    for textspan_Sentence in xmi.getElementsByTagName("textspan:Sentence"):
        Sentence.append([textspan_Sentence.getAttribute("begin"), textspan_Sentence.getAttribute("end")])
    for syntax_BaseToken in xmi.getElementsByTagName("syntax:BaseToken"):
        BaseToken.append([syntax_BaseToken.getAttribute("begin"), syntax_BaseToken.getAttribute("end")])
    for typesystem_PlatoNamedEntityUIMA in xmi.getElementsByTagName("typesystem:PlatoNamedEntityUIMA"):
        PlatoNamedEntityUIMA.append([typesystem_PlatoNamedEntityUIMA.getAttribute("begin"), typesystem_PlatoNamedEntityUIMA.getAttribute("end"),
                                     typesystem_PlatoNamedEntityUIMA.getAttribute("semanticTag"), typesystem_PlatoNamedEntityUIMA.getAttribute("xmi:id")])
    for typesystem_PlatoRelationUIMA in xmi.getElementsByTagName("typesystem:PlatoRelationUIMA"):
        PlatoRelationUIMA.append([typesystem_PlatoRelationUIMA.getAttribute("entFrom"), typesystem_PlatoRelationUIMA.getAttribute("entTo"),
                                 typesystem_PlatoRelationUIMA.getAttribute("semanticTag")])

    # save data
    if reload == True:
        if os.path.exists(xml2json): os.remove(xml2json)
        with codecs.open(xml2json, 'a+', encoding='utf-8') as fw:
            fw.write(json.dumps(text)+'\n'+json.dumps(PlatoNamedEntityUIMA)+'\n'+json.dumps(PlatoRelationUIMA)+'\n')
            fw.close()

    NamedEntity = modifyTag(PlatoNamedEntityUIMA)

    return text, NamedEntity, PlatoRelationUIMA

def modifyTag(PlatoNamedEntityUIMA):
    results = []
    for item in PlatoNamedEntityUIMA:
        result = item
        entity = item[2]
        if entity == '不良反应-诊断' or entity == '不良反应-疾病':
            result[2] = '不良反应-引发疾病'
        elif entity == '成分-复方-成分-含量' or entity == '成分-复方-成分':
            result[2] = '成分-复方成分'
        elif entity == '注意事项-人群' or entity == '注意事项-疾病相关':
            result = []
        elif entity == '用法用量-每日剂量-低值' or entity == '用法用量-每日剂量-高值':
            result[2] = '用法用量-每日剂量'
        elif entity == '用法用量-每次剂量-低值' or entity == '用法用量-每次剂量-高值':
            result[2] = '用法用量-每次剂量'
        elif entity == '用法用量-疾病' or entity == '用法用量-疾病状态-低值':
            result = []
        elif entity == '用法用量-给药频次-低值' or entity == '用法用量-给药频次-高值':
            result[2] = '用法用量-给药频次'
        elif entity == '用法用量-起始剂量-低值' or entity == '用法用量-起始剂量-高值':
            result[2] = '用法用量-起始剂量'
        elif entity == '用法用量-疾病-低值' or entity == '用法用量-疾病-高值':
            result[2] = '用法用量-疾病'
        elif entity == '药物禁忌-人群':
            result = []
        elif entity == '适应症-症状' or entity == '适应症-诊断根据':
            result = []
        else:
            pass

        if result != []:
            results.append(result)
    return results

def getType(source_dir, type_file, reload=False):
    """
    store all types and relevant counts of NER and relation and save in 'temp/'

    :param source_dir: the directory that store xml files
    :param type_file:save NER types and relation types
    :param reload: if reload is True, the save file will be constructed newly

    :return NERType: dictionary, the tag types of NER types
    :return relationType: dictionary, the tag types of relation types
    """
    NERType, relationType = {}, {}

    for xml_file in os.listdir(source_dir):
        text, NamedEntity, Relation = loadOneXml(source_dir+xml_file, hp.xml2json, reload)
        for ner in NamedEntity:
            begin, end, type = int(ner[0]), int(ner[1]), ner[2]
            NERType[type] = 1 if type not in NERType else NERType[type] + 1
        for rela in Relation:
            ent_from, ent_to, type = rela[0], rela[1], rela[2]
            relationType[type] = 1 if type not in relationType else relationType[type] + 1

    # save data
    if reload == True:
        if os.path.exists(type_file): os.remove(type_file)
        with codecs.open(type_file, 'w', 'utf-8') as fw:
            fw.write(json.dumps(NERType)+'\n'+json.dumps(relationType)+'\n')
            fw.close()
    # a = (sorted(NERType.items(), key=lambda item:item[1], reverse=True))
    # b = []
    # for i in range(len(a)):
    #     b.append(a[i][0])
    # print(b)
    return NERType, relationType


def xml2NER(source_dir, ner_file, reload=False):
    """
    transfer json format to the format of NER inputs

    :param source_dir: the directory that store xml files
    :param ner_file:save processed NER format data
    :param reload: if reload is True, the save file will be constructed newly

    :return S: list, the processed and split sentences
    :return T: list, correspond tags of the processed and split sentences
    """
    text2tag, tag2label = hp.text2tag, hp.tag2label
    sentences, tags = [], []
    for xml_file in os.listdir(source_dir):
        text, NamedEntity, Relation = loadOneXml(source_dir+xml_file, hp.xml2json, reload)
        sentence, tag = [], []
        for t in text:
            sentence.append(t)
        for i in range(len(sentence)):
            tag.append('O')
        for ner in NamedEntity:
            begin, end, type = int(ner[0]), int(ner[1]), ner[2]
            end = end-1
            # store as "BIESO" in tags
            if begin == end:
                tag[begin] = "S-"+ner[2]
            elif begin + 1 == end:
                tag[begin] = "B-"+ner[2]
                tag[end] = "E-" + ner[2]
            elif begin + 2 == end:
                tag[begin] = "B-" + ner[2]
                tag[begin+1] = "I-" + ner[2]
                tag[end] = "E-" + ner[2]
            else:
                tag[begin] = "B-"+ner[2]
                for i in range(begin+1, end-1):
                    tag[i] = "I-" + ner[2]
                if text[end] in ['，', ',', '。', '：', ':', '；', ';', '、', '）', ' ', '\n']:
                    tag[end-1] = "E-" + ner[2]
                else:
                    tag[end - 1] = "I-" + ner[2]
                    tag[end] = "E-" + ner[2]

        sentences.extend(sentence)
        tags.extend(tag)
    print("处理前句子长度：{} ； 标注长度：{}".format(len(sentences), len(tags)))
    sentences, tags = processSentences(sentences, tags)
    print("处理后句子长度：{} ； 标注长度：{}".format(len(sentences), len(tags)))

    # split sentences
    S,T = splitSentence(sentences, tags)

    #save
    if reload == True:
        if os.path.exists(ner_file): os.remove(ner_file)
        with codecs.open(ner_file, 'a+', encoding='utf-8') as fw:
            for (sent, tag) in zip(S, T):
                for (char, t) in zip(sent, tag):
                    fw.write(char + ' ' + t +'\n')
                fw.write('\n')
            fw.close()

    return S, T


def processSentences(sentences, tags):
    """
    process data, delete trashy chars and corresponding tags

    :param sentences: all the sentences
    :param tags:correspond tags of sentences

    :return sentences: the processed sentences
    :return tags: list, correspond tags of the processed sentences
    """
   # sentences = re.split('(。|！|\!|\.|？|\?)', paragraph)  # 保留分割符
    # delete '【' and '】' and their middles
    # while('【' in sentences and '】' in sentences):
    #     begin = sentences.index('【')
    #     end = sentences.index('】')
    #     if end > begin and end-begin < 15:
    #         for i in range(begin, end+1):
    #             sentences.pop(begin)
    #             tags.pop(begin)
    #     elif end < begin:
    #         sentences.pop(end)
    #         tags.pop(end)
    #     else:
    #         print("{}-{} 有多余的【】".format(begin, end))
    #         break

    # delete '\r', '\n', '?', ' ' in sentences
    list = ['\r', '\n', '?', ' ', '①', '②', '']
    for c in list:
        while (c in sentences):
            index = sentences.index(c)
            sentences.pop(index)
            tags.pop(index)

    # delete item numbers, include '（1）' and '1' and '1.' and '1、', there is "。" before them
    start = 0
    while('（' in sentences and '）' in sentences):
        try:
            begin = sentences.index('（', start)
            end = sentences.index('）', start)
            if end > begin and end - begin == 2 or end - begin == 3 and sentences[begin + 1].isdigit() and sentences[begin-1] == '。':
                for i in range(begin, end + 1):
                    sentences.pop(begin)
                    tags.pop(begin)
            else:
                start = end + 1
        except:
            break

    i = 0
    while(i+2 < len(sentences)):
        if sentences[i].isdigit() and sentences[i-1] == '。' and sentences[i+1] in ['.', '．','、'] and is_Chinese(sentences[i+2]):
            for j in range(2):
                sentences.pop(i)
                tags.pop(i)
        elif sentences[i].isdigit() and sentences[i-1] == '。' and is_Chinese(sentences[i+1]):
            sentences.pop(i)
            tags.pop(i)
        else:
            i += 1

    return sentences, tags


def splitSentence(sentences, tags):
    """
    split sentenses by "。" and save in S,T to yield train data and test data

    :param sentences: all the processed sentences
    :param tags:correspond tags of  processed sentences

    :return S: list, the split sentences
    :return T: list, correspond tags of the split sentences
    """
    S, T = [], []
    start = 0
    for i in range(len(sentences)):
        if i - start < 60:
            if sentences[i] in ['。']:
                sent, tag = [], []
                for j in range(start, i+1):
                    sent.append(sentences[j])
                    tag.append(tags[j])
                start = i + 1
                S.append(sent)
                T.append(tag)
        else:
            if sentences[i] in ['。', '，']:
                sent, tag = [], []
                for j in range(start, i + 1):
                    sent.append(sentences[j])
                    tag.append(tags[j])
                start = i + 1
                S.append(sent)
                T.append(tag)
    return S, T



# NERType, relationType = getType(hp.source_dir, hp.type_file, reload=True)
# print(NERType)
# print(relationType)

#_, _ = xml2NER(hp.source_dir, hp.ner_file, reload=True)
