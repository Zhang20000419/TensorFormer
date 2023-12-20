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
        feature_field = [text_field, visual_field, acoustic_field]
        if not pre_downloaded:
            # 在windows上需要先将site-packages\mmsdk\mmdatasdk\computational_sequence中的
            # destination=os.path.join(destination,resource.split(os.sep)[-1])
            # 改为destination=os.path.join(destination,resource.split('/')[-1])
            self.dataset = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, download_directory)
        else:
            self.dataset = mmdatasdk.mmdataset(download_directory)
            print('CMU-MOSI dataset has been downloaded before.\n', end='')
        if not pre_aligned:
            self.dataset = mmdatasdk.mmdataset({feature: os.path.join(download_directory, feature) + '.csd' for feature in feature_field})

            def myavg(intervals, features):
                return numpy.mean(features, axis=0)
            self.dataset.align(text_field, collapse_functions=[myavg])
            if not os.path.exists(labels_directory) or len(os.listdir(labels_directory)) == 0:
                self.dataset.add_computational_sequences(mmdatasdk.cmu_mosi.labels, labels_directory)
            else:
                # 需要修改
                self.dataset.add_computational_sequences(os.path.join(labels_directory, 'CMU_MOSI_Opinion_Labels.csd'), destination=None)

            self.dataset.add_computational_sequences(mmdatasdk.cmu_mosi.labels, labels_directory)
            self.dataset.align(feature_field, './aligned-cmu-mosi')







