import tensorflow as tf
import os, gc
from main import model_path, paths, config, embeddings, word2id
from model import BiLSTM_CRF


# result_path = 'ner_demo'
def predictNER(direction):
    sentences = direction.split('。')
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(embeddings, word2id, config, paths)
    model.buildGraph()
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        print("============= demo ==============")
        saver.restore(sess, paths['model_path'])

        results = []
        for sents in sentences:
            sents = list(sents.strip())
            data = [(sents, ['O']*len(sents))]
            tags = model.demoOne(sess, data)
            # fw = open(result_path, 'a+', encoding='utf-8')
            # for sent, tag in zip(sents, tags):
            #     fw.write(str(sent) + ' ' + str(tag) + '\n')
            # fw.write('\n')
            # fw.close()
            results.append([sents, tags])
        return results

string = ''
direction = '对头孢菌素类药物过敏者、严重肝功能障碍者、中度或严重肾功能障碍者及有哮喘、湿疹、枯草热、荨麻疹等过敏性疾病史者慎用。' \
            '本品适用于革兰氏阳性菌引起的下列各种感染性疾病：扁桃体炎、化脓性中耳炎、鼻窦炎等。急性支气管炎、慢性支气管炎急性发作、肺炎、肺脓肿和支气管扩张合并感染等。'
print(predictNER(direction))