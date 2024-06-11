import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import torch.nn.functional as F
import  numpy
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (256,256))
            labelss = np.array((labels).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (256,256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        # #改
        # Final_FA =  self.FA / (((256 * 256) * img_num)-self.target)

        Final_PD =  self.PD /self.target


        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class PD_FA_pix():
    def __init__(self, nclass, bins):
        super(PD_FA_pix, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.FA = torch.zeros(self.bins + 1)
        self.PD = torch.zeros(self.bins + 1)
        self.total_negative_pixels = 0  # 真实负样本像素总数
        self.total_positive_pixels = 0  # 真实正样本像素总数

    def update(self, preds, labels):
        preds = preds.cpu()
        labels = labels.cpu()
        for iBin in range(self.bins + 1):
            score_thresh = iBin * (255 / self.bins)
            preds_binary = (preds > score_thresh).type(torch.int64)
            labels_binary = labels.type(torch.int64)

            true_positives = (preds_binary == 1) & (labels_binary == 1)
            false_positives = (preds_binary == 1) & (labels_binary == 0)
            true_negatives = (preds_binary == 0) & (labels_binary == 0)
            false_negatives = (preds_binary == 0) & (labels_binary == 1)

            # 累加计算
            self.PD[iBin] += true_positives.sum()
            self.FA[iBin] += false_positives.sum()
            self.total_positive_pixels += labels_binary.sum()
            self.total_negative_pixels += (labels_binary == 0).sum()

    def get(self):
        Final_FA = self.FA.float() / self.total_negative_pixels
        Final_PD = self.PD.float() / self.total_positive_pixels

        return Final_FA.numpy(), Final_PD.numpy()

    def reset(self):
        self.FA = torch.zeros(self.bins + 1)
        self.PD = torch.zeros(self.bins + 1)
        self.total_negative_pixels = 0
        self.total_positive_pixels = 0


class PD_FA_gai():
    def __init__(self, nclass, bins):
        super(PD_FA_gai, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = iBin * (255 / self.bins)
            preds_binary = np.array((preds > score_thresh).cpu()).astype('int64').reshape((256, 256))
            labels_binary = np.array(labels.cpu()).astype('int64').reshape((256, 256))

            pred_labels = measure.label(preds_binary, connectivity=2)
            true_labels = measure.label(labels_binary, connectivity=2)

            pred_props = measure.regionprops(pred_labels)
            true_props = measure.regionprops(true_labels)

            self.target[iBin] += len(true_props)

            matched_pred_indices = []

            for true_prop in true_props:
                true_centroid = np.array(true_prop.centroid)
                for pred_index, pred_prop in enumerate(pred_props):
                    if pred_index in matched_pred_indices:
                        continue
                    pred_centroid = np.array(pred_prop.centroid)
                    if np.linalg.norm(pred_centroid - true_centroid) < 3:
                        matched_pred_indices.append(pred_index)
                        break

            unmatched_pred_areas = [pred_props[i].area for i in range(len(pred_props)) if i not in matched_pred_indices]
            self.FA[iBin] += np.sum(unmatched_pred_areas)
            self.PD[iBin] += len(matched_pred_indices)

    def get(self, img_num):
        Final_FA = self.FA / ((256 * 256) * img_num)
        Final_PD = self.PD / self.target
        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)

class PD_FA_pix2():
    def __init__(self, nclass, bins):
        self.bins = bins
        self.TP = np.zeros(self.bins + 1)
        self.FP = np.zeros(self.bins + 1)
        self.TN = np.zeros(self.bins + 1)
        self.FN = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        preds = preds.cpu()
        labels = labels.cpu()
        for iBin in range(self.bins + 1):
            # Adjust threshold range according to your preds distribution
            score_thresh = iBin * (255 / self.bins)
            preds_binary = (preds > score_thresh).numpy().astype('int64')
            preds_binary = np.reshape(preds_binary, (256, 256))
            labels_np = labels.numpy().astype('int64')
            labels_np = np.reshape(labels_np, (256, 256))

            TP = np.sum((preds_binary == 1) & (labels_np == 1))
            FP = np.sum((preds_binary == 1) & (labels_np == 0))
            TN = np.sum((preds_binary == 0) & (labels_np == 0))
            FN = np.sum((preds_binary == 0) & (labels_np == 1))

            self.TP[iBin] += TP
            self.FP[iBin] += FP
            self.TN[iBin] += TN
            self.FN[iBin] += FN

    def get(self):
        TPR = self.TP / (self.TP + self.FN + np.finfo(float).eps)
        FPR = self.FP / (self.FP + self.TN + np.finfo(float).eps)
        # Handle division by zero by adding a small epsilon value
        TPR = np.nan_to_num(TPR).tolist()
        FPR = np.nan_to_num(FPR).tolist()
        return FPR, TPR

    def reset(self):
        self.TP = np.zeros(self.bins + 1)
        self.FP = np.zeros(self.bins + 1)
        self.TN = np.zeros(self.bins + 1)
        self.FN = np.zeros(self.bins + 1)


class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

#添加　计算niou
class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        inter_arr, union_arr = self.batch_intersection_union(preds, labels,
                                                             self.nclass, self.score_thresh)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        """mIoU"""
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        # predict = (F.sigmoid(output).detach().numpy() > score_thresh).astype('int64') # P
        # target = target.detach().numpy().astype('int64') # T
        # intersection = predict * (predict == target) # TP

        predict = (output > 0).float()
        if len(target.shape) == 3:
            target = np.expand_dims(target.float(), axis=1)
        elif len(target.shape) == 4:
            target = target.float()
        else:
            raise ValueError("Unknown target dimension")
        intersection = predict * ((predict == target).float())
        predict = predict.cpu()
        target = target.cpu()
        intersection = intersection.cpu()

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr
#



def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

