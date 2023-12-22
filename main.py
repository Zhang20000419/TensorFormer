from CMU_MOSI_DataLoader import CMU_MOSI_Dataset

if __name__ == '__main__':
    # 是否提前下载了CMU-MOSI数据集
    pre_downloaded = True
    pre_aligned = True
    cmu_dataloader = CMU_MOSI_Dataset(pre_downloaded, pre_aligned)
    print(cmu_dataloader.dataset.computational_sequences.keys(), end='')
    # print(cmu_dataloader.dataset.computational_sequences['CMU_MOSI_Visual_Facet_41'][55], end='')
    print(list(cmu_dataloader.dataset.computational_sequences['FACET_4.1'].keys())[55], end='')




