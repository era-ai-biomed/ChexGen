# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.datasets.base_dataset import BaseDataset as OLD_BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.multi_label import \
    MultiLabelDataset as OLD_MultiLabelDataset
from mmcls.models.losses import accuracy

from medfmc.core.evaluation import AUC_multiclass, AUC_multilabel

import numpy as np
import torch
import warnings

def average_performance(pred, target, thr=None, k=None):
    """Calculate CP, CR, CF1, OP, OR, OF1, WP, WR, WF1, where:
    C stands for per-class (macro) average,
    O stands for overall average,
    W stands for weighted average,
    P stands for precision,
    R stands for recall,
    F1 stands for F1-score.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 
            1 stands for positive examples, 0 stands for negative examples and 
            -1 stands for difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1, WP, WR, WF1)
    """
    # 转换为numpy数组
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or np.ndarray')
    
    # 设置默认阈值，并给出警告
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of top-k.')

    # 保证预测和真实标签形状一致
    assert pred.shape == target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    # 将难例(-1)视作负例
    target[target == -1] = 0

    # 根据阈值或top-k确定预测为正的指标
    if thr is not None:
        # 如果预测值不低于阈值，则认为该标签预测为正
        pos_inds = pred >= thr
    else:
        # top-k策略：每个样本选取得分最高的k个标签为正
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1

    # 计算 true positive, false positive 和 false negative
    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1

    # per-class precision 和 recall
    precision_class = tp.sum(axis=0) / np.maximum(tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(tp.sum(axis=0) + fn.sum(axis=0), eps)

    # 宏平均（C: per-class）
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)

    # 全局平均（O: overall）
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)

    # 加权平均（W: weighted）
    # 计算每个类别的 support (真实正例数量)
    support = target.sum(axis=0)
    total_support = np.maximum(np.sum(support), eps)
    WP = np.sum(precision_class * support) / total_support * 100.0
    WR = np.sum(recall_class * support) / total_support * 100.0
    # 计算每个类别的F1 score
    f1_class = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps)
    WF1 = np.sum(f1_class * support) / total_support * 100.0

    return CP, CR, CF1, OP, OR, OF1, WP, WR, WF1



def average_precision(pred, target):
    r"""Calculate the average precision for a single class.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).

    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap

def mAP(pred, target):
    """Calculate the mean average precision with respect of classes.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.

    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
        print('{:.2f}'.format(ap[k] * 100))
    mean_ap = ap.mean() * 100.0
    return mean_ap

class BaseDataset(OLD_BaseDataset):

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support',
            'AUC_multiclass'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        print("using thres", metric_options.get('thrs'))
        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'AUC_multiclass' in metrics:
            AUC_value = AUC_multiclass(results, gt_labels)
            eval_results['AUC_multiclass'] = AUC_value

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results


class MultiLabelDataset(OLD_MultiLabelDataset):

    def evaluate(self,
                 results,
                 metric='AUC_multilabel',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1', 'AUC_multilabel.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        if metric_options is None or metric_options == {}:
            metric_options = {'thr': 0.5}

        print("metric_options", metric_options)

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1','WP', 'WR', 'WF1', 'AUC_multilabel'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if 'AUC_multilabel' in metrics:
            AUC_value = AUC_multilabel(results, gt_labels)
            eval_results['AUC_multilabel'] = AUC_value
        if len(set(metrics) - {'mAP'}) != 0:
            performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'WP', 'WR', 'WF1']
            performance_values = average_performance(results, gt_labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results


@DATASETS.register_module()
class Chest19(MultiLabelDataset):

    CLASSES = [
        'pleural_effusion', 'nodule', 'pneumonia', 'cardiomegaly',
        'hilar_enlargement', 'fracture_old', 'fibrosis',
        'aortic_calcification', 'tortuous_aorta', 'thickened_pleura', 'TB',
        'pneumothorax', 'emphysema', 'atelectasis', 'calcification',
        'pulmonary_edema', 'increased_lung_markings', 'elevated_diaphragm',
        'consolidation'
    ]

    def __init__(self, **kwargs):
        super(Chest19, self).__init__(**kwargs)

    def load_annotations(self):
        if isinstance(self.ann_file, str):
            self.ann_file = [self.ann_file]
        if isinstance(self.data_prefix, str):
            self.data_prefix = [self.data_prefix]
        
        data_infos = []
        for i, ann_file in enumerate(self.ann_file):
            with open(ann_file) as f:
                samples = [x.strip() for x in f.readlines()]
                for item in samples:
                    filename, imglabel = item.split(' ')
                    gt_label = np.asarray(
                        list(map(int, imglabel.split(','))), dtype=np.int8)

                    info = {'img_prefix': self.data_prefix[i]}
                    info['img_info'] = {'filename': filename}
                    info['gt_label'] = gt_label

                    data_infos.append(info)
        # data_infos = []
        # with open(self.ann_file) as f:
        #     samples = [x.strip() for x in f.readlines()]
        #     for item in samples:
        #         filename, imglabel = item.split(' ')
        #         gt_label = np.asarray(
        #             list(map(int, imglabel.split(','))), dtype=np.int8)

        #         info = {'img_prefix': self.data_prefix}
        #         info['img_info'] = {'filename': filename}
        #         info['gt_label'] = gt_label

        #         data_infos.append(info)

        return data_infos

@DATASETS.register_module()
class MIMIC(MultiLabelDataset):
    CLASSES = [
        "no finding", "enlarged cardiomediastinum", "cardiomegaly", "airspace opacity", "lung lesion", "edema",
        "consolidation", "pneumonia", "atelectasis", "pneumothorax", "pleural effusion", "pleural other",
        "fracture", "support devices"]

    def __init__(self, **kwargs):
        super(MIMIC, self).__init__(**kwargs)

    def load_annotations(self):
            if isinstance(self.ann_file, str):
                self.ann_file = [self.ann_file]
            if isinstance(self.data_prefix, str):
                self.data_prefix = [self.data_prefix]
            
            data_infos = []
            for i, ann_file in enumerate(self.ann_file):
                with open(ann_file) as f:
                    samples = [x.strip() for x in f.readlines()]
                    for item in samples:
                        filename, imglabel = item.split(' ')
                        gt_label = np.asarray(
                            list(map(int, imglabel.split(','))), dtype=np.int8)
                        info = {'img_prefix': self.data_prefix[i]}
                        info['img_info'] = {'filename': filename}
                        info['gt_label'] = gt_label
                        data_infos.append(info)

            return data_infos


@DATASETS.register_module()
class VINDRCXR(MultiLabelDataset):
    CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding"
    ]

    def __init__(self, **kwargs):
        super(VINDRCXR, self).__init__(**kwargs)

    def load_annotations(self):
            if isinstance(self.ann_file, str):
                self.ann_file = [self.ann_file]
            if isinstance(self.data_prefix, str):
                self.data_prefix = [self.data_prefix]
            
            data_infos = []
            for i, ann_file in enumerate(self.ann_file):
                with open(ann_file) as f:
                    samples = [x.strip() for x in f.readlines()]
                    for item in samples:
                        filename, imglabel = item.split(' ')
                        gt_label = np.asarray(
                            list(map(int, imglabel.split(','))), dtype=np.int8)
                        info = {'img_prefix': self.data_prefix[i]}
                        info['img_info'] = {'filename': filename}
                        info['gt_label'] = gt_label
                        data_infos.append(info)

            return data_infos


@DATASETS.register_module()
class Endoscopy(MultiLabelDataset):

    CLASSES = ['ulcer', 'erosion', 'polyp', 'tumor']

    def __init__(self, **kwargs):
        super(Endoscopy, self).__init__(**kwargs)

    def load_annotations(self):

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename = item[:-8]
                imglabel = item[-7:]
                gt_label = np.asarray(
                    list(map(int, imglabel.split(' '))), dtype=np.int8)

                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = gt_label

                data_infos.append(info)

        return data_infos


@DATASETS.register_module()
class Colon(BaseDataset):

    CLASSES = ['negtive', 'positive']

    def __init__(self, **kwargs):
        super(Colon, self).__init__(**kwargs)

    def load_annotations(self):

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename = item[:-2]
                imglabel = int(item[-1:])
                gt_label = np.array(imglabel, dtype=np.int64)

                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = gt_label

                data_infos.append(info)

        return data_infos
