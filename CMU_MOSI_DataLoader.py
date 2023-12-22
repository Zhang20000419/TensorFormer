import os.path

from mmsdk import mmdatasdk
from torch.utils.data import Dataset, DataLoader
import numpy


class CMU_MOSI_Dataset(Dataset):
    """
    CMU-MOSI数据集的数据加载器
    :parameter:pre_downloaded:是否提前下载了CMU-MOSI数据集
    :parameter:pre_aligned:是否提前对CMU-MOSI数据集进行了对齐
    :parameter:directory:CMU-MOSI数据集的存放路径,默认为'./cmu-mosi'
    """
    def __init__(self, pre_downloaded, pre_aligned, download_directory='.\\cmu-mosi', alignment_directory='.\\aligned-cmu-mosi', labels_directory='.\\labels-cmu-mosi'):
        # 要对齐的特征
        visual_field = 'CMU_MOSI_Visual_Facet_41'
        acoustic_field = 'CMU_MOSI_COVAREP'
        text_field = 'CMU_MOSI_TimestampedWordVectors'
        label_field = 'CMU_MOSI_Opinion_Labels'
        feature_field = [text_field, visual_field, acoustic_field]
        if not pre_downloaded and pre_aligned:
            print('parameter error: pre_aligned must be False when pre_downloaded is False.\n', end='')
            return
        if pre_downloaded and pre_aligned:
            self.dataset = mmdatasdk.mmdataset(alignment_directory)
            print('CMU-MOSI dataset has been downloaded and aligned before.\n', end='')

        elif not pre_downloaded:
            # 在windows上需要先将site-packages\mmsdk\mmdatasdk\computational_sequence中的
            # destination=os.path.join(destination,resource.split(os.sep)[-1])
            # 改为destination=os.path.join(destination,resource.split('/')[-1])
            self.dataset = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, download_directory)
            print('CMU-MOSI dataset has been downloaded.\n', end='')
        else:
            self.dataset = mmdatasdk.mmdataset(download_directory)
            print('CMU-MOSI dataset has been downloaded before.\n', end='')
        # 对齐
        if not pre_aligned:
            self.dataset = mmdatasdk.mmdataset({feature: os.path.join(download_directory, feature) + '.csd' for feature in feature_field})

            def myavg(intervals, features):
                return numpy.mean(features, axis=0)
            # 按照text_field对齐
            self.dataset.align(text_field, collapse_functions=[myavg])
            # 如果没有提前下载标签，就下载标签
            if not os.path.exists(labels_directory) or len(os.listdir(labels_directory)) == 0:
                self.dataset.add_computational_sequences({labels_directory: mmdatasdk.cmu_mosi.labels['Opinion Segment Labels']}, labels_directory)
            else:
                self.dataset.add_computational_sequences({label_field: os.path.join(labels_directory, label_field + '.csd')}, destination=None)
            # 按照label_field对齐
            self.dataset.align(label_field, replace=True)

            # 保存对齐后的数据集
            deploy_files = {x: x for x in self.dataset.computational_sequences.keys()}
            self.dataset.deploy(alignment_directory, deploy_files)
            print('CMU-MOSI dataset has been aligned and writen into directory:%s' % {alignment_directory}, end='')
        else:
            self.dataset = mmdatasdk.mmdataset(alignment_directory)






