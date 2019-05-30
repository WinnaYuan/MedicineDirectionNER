import os

def conlleval(model_predict, label_path, metric_path):
    """
    use "conlleval_rev.pl" to evaluate the results

    :param label_predict: list, predicted labels
    :param label_path: store text results
    :param metric_path: store metric results
    :return: string , metric result
    """
    eval_perl = "./conlleval_rev.pl"
    with open(label_path ,'a+', encoding='utf-8') as fw:
        for sent_result in model_predict:
            for chara, tag, tag_pred in sent_result:
                if tag == 'O': tag = '0'
                fw.write(chara + ' ' + tag + ' ' + str(tag_pred) + '\n')
            fw.write('\n')
        fw.close()

    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    fr = open(metric_path, 'r', encoding='utf-8')
    metric = [line.strip() for line in fr.readlines()]

    return metric