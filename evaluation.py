import numpy as np
import os
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from viz import *

import config as c
from model import *
from utils import *

from sklearn.metrics import precision_recall_curve
import csv

import os

# F1 evalution code from https://github.com/caoyunkang/WinClip

def calculate_f1_max(gt, scores):
    precision, recall, thresholds = precision_recall_curve(gt, scores)
    a = 2 * precision * recall
    b = precision + recall
    f1s = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    index = np.argmax(f1s)
    max_f1 = f1s[index]
    threshold = thresholds[index]
    return max_f1, threshold


def localize(image, depth, st_pixel, labels, fg, mask, batch_ind, img_class, img_id_inclass):
    for i in range(fg.shape[0]):
        fg_i = t2np(fg[i, 0])
        depth_viz = t2np(depth[i, 0])
        depth_viz[fg_i == 0] = np.nan
        viz_maps(t2np(image[i]), depth_viz, t2np(mask[i, 0]), t2np(st_pixel[i]), fg_i,
                 img_class[i] + '_' + img_id_inclass[i], norm=True, enable_pixel_eval=False)


def evaluate(test_loader, enable_pixel_eval=False):
    student = Model(nf=not c.asymmetric_student, channels_hidden=c.channels_hidden_student)
    student = load_weights(student, 'student')

    teacher = Model()
    teacher = load_weights(teacher, 'teacher')

    up = torch.nn.Upsample(size=None, scale_factor=c.depth_len // c.map_len, mode='bicubic',
                           align_corners=False)

    test_labels = list()
    mean_st = list()
    max_st = list()

    score_maps = list()
    gt_masks = list()

    image_classes = list()
    image_ids_inclass = list()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            depth, fg, labels, image, features, mask, image_class, image_id_inclass = data
            depth, fg, image, features, mask = to_device([depth, fg, image, features, mask])
            fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

            img_in = features if c.pre_extracted else image
            fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)

            z_t, jac_t = teacher(img_in, depth)
            z, jac = student(img_in, depth)

            st_loss = get_st_loss(z_t, z, fg_down, per_sample=True)
            st_pixel = get_st_loss(z_t, z, fg_down, per_pixel=True)

            if c.eval_mask:
                st_pixel = st_pixel * fg_down[:, 0]
            st_pixel = up(st_pixel[:, None])[:, 0]

            mean_st.append(t2np(st_loss))
            max_st.append(np.max(t2np(st_pixel), axis=(1, 2)))
            test_labels.append(labels)
            gt_masks.append(t2np(mask).flatten())
            score_maps.append(t2np(st_pixel).flatten())

            if c.localize:
                localize(image, depth, st_pixel, labels, fg, mask, i, image_class, image_id_inclass)

            image_classes.extend(image_class)
            image_ids_inclass.extend(image_id_inclass)

    mean_st = np.concatenate(mean_st)
    max_st = np.concatenate(max_st)

    gt_masks = np.concatenate(gt_masks)
    score_maps = np.concatenate(score_maps)

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    mean_st_auc = roc_auc_score(is_anomaly, mean_st)
    max_st_auc = roc_auc_score(is_anomaly, max_st)

    # visualize roc curve
    viz_roc(mean_st, is_anomaly, name='mean')
    viz_roc(max_st, is_anomaly, name='max')

    # visualize histogram
    compare_histogram(mean_st, is_anomaly, log=True, name='mean')
    compare_histogram(max_st, is_anomaly, log=True, name='max')

    # ccompute image-level f1 score using mean or max over anomaly maps.
    img_f1_mean, img_threshold_mean = calculate_f1_max(is_anomaly, mean_st)
    img_f1_max, img_threshold_max = calculate_f1_max(is_anomaly, max_st)
    img_f1 = img_f1_mean if img_f1_mean > img_f1_max else img_f1_max
    img_threshold = img_threshold_mean if img_f1_mean > img_f1_max else img_threshold_max

    # choose the criterion resulting in the highes f1 score. 
    if img_f1_mean > img_f1_max:
        predictions = np.array([0 if l < img_threshold_mean else 1 for l in mean_st])
    else:
        predictions = np.array([0 if l < img_threshold_max else 1 for l in mean_st])

    # write image-wise evaluation results in a csv file.
    with open('evaluation_results.csv', 'a', newline='') as csv_file:
        if img_f1_mean > img_f1_max:
            csv_file.write('"using the mean over maps as AD criterion"\n')
        else:
            csv_file.write('"using the max over maps as AD criterion"\n')
    img_infos = list()
    img_infos.append(['Defects type', 'Image Nr.', 'Groud truth', 'Prediction'])
    for i_class, id_inclass, gt_anomaly, pred in zip(image_classes, image_ids_inclass, is_anomaly, predictions):
        gt_anomaly = 'good' if gt_anomaly == 0 else 'anomalous'
        pred = 'good' if pred == 0 else 'anomalous'
        img_infos.append([i_class, id_inclass, gt_anomaly, pred])

    with open('evaluation_results.csv', 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(img_infos)


    # print out display
    if enable_pixel_eval:
        pixel_auc = roc_auc_score(gt_masks, score_maps)

        print('AUROC %\t mean over maps: {:.2f} \t max over maps: {:.2f} \t pixel: {:.2f}'.format(mean_st_auc * 100, max_st_auc * 100, pixel_auc * 100))
        print('F1 score % \t mean over maps: {:.2f} \t max over maps: {:.2f}'.format(img_f1_mean * 100, img_f1_max * 100))
        print('Image-level AD threshold \t mean over maps: {:.2f} \t max over maps: {:.2f} \n'.format(img_threshold_mean, img_threshold_max))

        viz_roc(score_maps, gt_masks, name='pixel')
        return mean_st_auc, max_st_auc, img_f1, pixel_auc

    print('AUROC %\t mean over maps: {:.2f} \t max over maps: {:.2f}'.format(mean_st_auc * 100, max_st_auc * 100))
    print('F1 score % \t mean over maps: {:.2f} \t max over maps: {:.2f}'.format(img_f1_mean * 100, img_f1_max * 100))
    print('Image-level AD threshold \t mean over maps: {:.2f} \t max over maps: {:.2f} \n'.format(img_threshold_mean, img_threshold_max))
    return mean_st_auc, max_st_auc, img_f1


if __name__ == "__main__":
    ENABLE_PIXEL_EVAL = False
    all_classes = [d for d in os.listdir(c.dataset_dir) if os.path.isdir(os.path.join(c.dataset_dir, d))]
    max_scores = list()
    mean_scores = list()
    f1_scores = list()
    pixel_scores = list()
    for i_c, cn in enumerate(all_classes):
        c.class_name = cn
        print('\nEvaluate class ' + c.class_name)
        train_set, test_set = load_datasets(get_mask=True)
        _, test_loader = make_dataloaders(train_set, test_set)
        if ENABLE_PIXEL_EVAL:
            mean_sc, max_sc, f1_sc, pixel_sc = evaluate(test_loader, enable_pixel_eval=True)
            pixel_scores.append(pixel_sc)
        else:
            mean_sc, max_sc, f1_sc = evaluate(test_loader, enable_pixel_eval=False)
        mean_scores.append(mean_sc)
        max_scores.append(max_sc)
        f1_scores.append(f1_sc)

        # log f1 score for each class
        if "mlflow_tracking_uri" in globals():
            mlflow.log_metric(f"f1-{cn}", f1_sc)

    mean_scores = np.mean(mean_scores) * 100
    max_scores = np.mean(max_scores) * 100
    f1_scores = np.mean(f1_scores) * 100
    if ENABLE_PIXEL_EVAL:
        pixel_scores = np.mean(pixel_scores) * 100
        print('\nmean AUROC % over all classes\n\tmean over maps: {:.2f} \t max over maps: {:.2f} \t pixel: {:.2f} \nmean F1 score % over all classes: {:.2f} '.format(mean_scores,
                                                                                                        max_scores, pixel_scores, f1_scores))
    else:
        print('\nmean AUROC % over all classes\n\tmean over maps: {:.2f} \t max over maps: {:.2f} \nmean F1 score % over all classes: {:.2f}'.format(mean_scores, max_scores, f1_scores))

    # log overall F1 score
    if "mlflow_tracking_uri" in globals():
        mlflow.log_metric("f1", f1_scores)
