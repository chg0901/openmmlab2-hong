from mmengine import Config

def modify(cfg):
    cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
    cfg.crop_size = (256, 256)
    cfg.model.data_preprocessor.size = cfg.crop_size
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head

    # 模型 decode/auxiliary 输出头，指定为类别个数
    cfg.model.decode_head.num_classes = 6
    cfg.model.auxiliary_head.num_classes = 6

    cfg.train_dataloader.batch_size = 8

    cfg.test_dataloader = cfg.val_dataloader

    # 结果保存目录
    cfg.work_dir = '../log/Watermelon'

    # 训练迭代次数
    cfg.train_cfg.max_iters = 3000
    # 评估模型间隔
    cfg.train_cfg.val_interval = 500
    # 日志记录间隔
    cfg.default_hooks.logger.interval = 100
    # 模型权重保存间隔
    cfg.default_hooks.checkpoint.interval = 500

    # 随机数种子
    cfg['randomness'] = dict(seed=0)


    # -------
    # 以下是数据集需要修改的参数
    cfg.data_root = '/home/zeyuan.yin/OpenMMLabCamp/homework-4/data/Watermelon87_Semantic_Seg_Mask'
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_root = cfg.data_root



if __name__ == '__main__':
    cfg = Config.fromfile('./configs/pspnet/pspnet_r50-d8_4xb2-40k_DubaiDataset.py')

    modify(cfg)
    cfg.dump('pspnet-Watermelon.py')
    print('done, config saved to pspnet-Watermelon.py')