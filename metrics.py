import torch
from dotted_dict import DottedDict


def compute_csc_metrics_by_file(predict_file, labels_file, show = False):
    # Compute F1-score
    detect_TP, detect_FP, detect_FN = 0, 0, 0
    correct_TP, correct_FP, correct_FN = 0, 0, 0
    detect_sent_TP, sent_P, sent_N, correct_sent_TP = 0, 0, 0, 0
    dc_TP, dc_FP, dc_FN = 0, 0, 0

    predicts = sorted(open(predict_file, "r", encoding="utf-8").readlines(), key=lambda l: l.split(",")[0].strip())
    labels = sorted(open(labels_file, "r", encoding="utf-8").readlines(), key=lambda l: l.split(",")[0].strip())

    for idx, (pred, actual) in enumerate(zip(predicts,
                                             labels)):  # copied from https://github.com/ACL2020SpellGCN/SpellGCN/blob/master/run_spellgcn.py
        pred_tokens = pred.strip().split(",")
        actual_tokens = actual.strip().split(",")
        # print(pred)
        # print(actual)
        assert (pred_tokens[0] == actual_tokens[0]) or (pred_tokens[0] in actual_tokens[0]) or (actual_tokens[0] in pred_tokens[0])

        pred_tokens = [x.strip() for x in pred_tokens[1:] ]
        actual_tokens = [x.strip() for x in actual_tokens[1:]]


        if len(pred_tokens) <= 1:
            pred_tokens = ["0"]
        if len(actual_tokens) <=1:
            actual_tokens = ["0"]

        if len(pred_tokens) % 2 == 1 and pred_tokens[-1] == "":
            del pred_tokens[-1]

        if len(actual_tokens) % 2 == 1 and actual_tokens[-1] == "":
            del actual_tokens[-1]

        detect_actual_tokens = [int(actual_token.strip(",")) \
                                for i, actual_token in enumerate(actual_tokens) if i % 2 == 0]
        correct_actual_tokens = [actual_token.strip(",") \
                                 for i, actual_token in enumerate(actual_tokens) if i % 2 == 1]
        detect_pred_tokens = [int(pred_token.strip(",")) \
                              for i, pred_token in enumerate(pred_tokens) if i % 2 == 0]
        _correct_pred_tokens = [pred_token.strip(",") \
                                for i, pred_token in enumerate(pred_tokens) if i % 2 == 1]

        # Postpreprocess for ACL2019 csc paper which only deal with last detect positions in test data.
        # If we wanna follow the ACL2019 csc paper, we should take the detect_pred_tokens to:

        max_detect_pred_tokens = detect_pred_tokens

        correct_pred_zip = zip(detect_pred_tokens, _correct_pred_tokens)
        correct_actual_zip = zip(detect_actual_tokens, correct_actual_tokens)

        if detect_pred_tokens[0] != 0:
            sent_P += 1
            if sorted(correct_pred_zip) == sorted(correct_actual_zip):
                correct_sent_TP += 1
        if detect_actual_tokens[0] != 0:
            if sorted(detect_actual_tokens) == sorted(detect_pred_tokens):
                detect_sent_TP += 1
            sent_N += 1

        if detect_actual_tokens[0] != 0:
            detect_TP += len(set(max_detect_pred_tokens) & set(detect_actual_tokens))
            detect_FN += len(set(detect_actual_tokens) - set(max_detect_pred_tokens))
        detect_FP += len(set(max_detect_pred_tokens) - set(detect_actual_tokens))

        correct_pred_tokens = []
        # Only check the correct postion's tokens
        for dpt, cpt in zip(detect_pred_tokens, _correct_pred_tokens):
            if dpt in detect_actual_tokens:
                correct_pred_tokens.append((dpt, cpt))

        correction_list = [actual.split(" ")[0].strip(",")]
        for dat, cpt in correct_pred_tokens:
            correction_list.append(str(dat))
            correction_list.append(cpt)

        correct_TP += len(set(correct_pred_tokens) & set(zip(detect_actual_tokens, correct_actual_tokens)))
        correct_FP += len(set(correct_pred_tokens) - set(zip(detect_actual_tokens, correct_actual_tokens)))
        correct_FN += len(set(zip(detect_actual_tokens, correct_actual_tokens)) - set(correct_pred_tokens))

        # Caluate the correction level which depend on predictive detection of BERT
        dc_pred_tokens = zip(detect_pred_tokens, _correct_pred_tokens)
        dc_actual_tokens = zip(detect_actual_tokens, correct_actual_tokens)
        dc_TP += len(set(dc_pred_tokens) & set(dc_actual_tokens))
        dc_FP += len(set(dc_pred_tokens) - set(dc_actual_tokens))
        dc_FN += len(set(dc_actual_tokens) - set(dc_pred_tokens))

    detect_precision = detect_TP * 1.0 / (detect_TP + detect_FP)
    detect_recall = detect_TP * 1.0 / (detect_TP + detect_FN)
    detect_F1 = 2. * detect_precision * detect_recall / ((detect_precision + detect_recall) + 1e-8)

    correct_precision = correct_TP * 1.0 / (correct_TP + correct_FP)
    correct_recall = correct_TP * 1.0 / (correct_TP + correct_FN)
    correct_F1 = 2. * correct_precision * correct_recall / ((correct_precision + correct_recall) + 1e-8)

    dc_precision = dc_TP * 1.0 / (dc_TP + dc_FP + 1e-8)
    dc_recall = dc_TP * 1.0 / (dc_TP + dc_FN + 1e-8)
    dc_F1 = 2. * dc_precision * dc_recall / (dc_precision + dc_recall + 1e-8)


    # Token-level metrics
    if show:
        print("detect_precision=%f, detect_recall=%f, detect_Fscore=%f" % (
            detect_precision, detect_recall, detect_F1))
        print("correct_precision=%f, correct_recall=%f, correct_Fscore=%f" % (
            correct_precision, correct_recall, correct_F1))
        print("dc_joint_precision=%f, dc_joint_recall=%f, dc_joint_Fscore=%f" % (dc_precision, dc_recall, dc_F1))

    detect_sent_precision = detect_sent_TP * 1.0 / (sent_P)
    detect_sent_recall = detect_sent_TP * 1.0 / (sent_N)
    detect_sent_F1 = 2. * detect_sent_precision * detect_sent_recall / (
            (detect_sent_precision + detect_sent_recall) + 1e-8)

    correct_sent_precision = correct_sent_TP * 1.0 / (sent_P)
    correct_sent_recall = correct_sent_TP * 1.0 / (sent_N)
    correct_sent_F1 = 2. * correct_sent_precision * correct_sent_recall / (
            (correct_sent_precision + correct_sent_recall) + 1e-8)

    # Sentence-level metrics
    if show:
        print("detect_sent_precision=%f, detect_sent_recall=%f, detect_Fscore=%f" % (
            detect_sent_precision, detect_sent_recall, detect_sent_F1))
        print("correct_sent_precision=%f, correct_sent_recall=%f, correct_Fscore=%f" % (
            correct_sent_precision, correct_sent_recall, correct_sent_F1))

    token_metric, sentence_metric = DottedDict(), DottedDict()
    token_metric.detect, token_metric.correct = DottedDict(), DottedDict()
    sentence_metric.detect, sentence_metric.correct = DottedDict(), DottedDict()
    token_metric.detect.precision, token_metric.detect.recall, token_metric.detect.f1 = detect_precision, detect_recall, detect_F1
    token_metric.correct.precision, token_metric.correct.recall, token_metric.correct.f1 = correct_precision, correct_recall, correct_F1
    sentence_metric.detect.precision, sentence_metric.detect.recall, sentence_metric.detect.f1 = detect_sent_precision, detect_sent_recall, detect_sent_F1
    sentence_metric.correct.precision, sentence_metric.correct.recall, sentence_metric.correct.f1 = correct_sent_precision, correct_sent_recall, correct_sent_F1
    metric = DottedDict()
    metric.token = token_metric
    metric.sentence = sentence_metric
    metric.f1 = metric.sentence.correct.f1
    return metric