from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3, UNet_URPC, UNet_CCT,BFDCNet2d_v1,DiceCENet2d_fuse,UNet_sdf
from networks.VNet import VNet, MCNet3d_v1, MCNet3d_v2, ECNet3d, DiceCENet3d,DiceCENet3d_fuse,DiceCENet3d_fuse_2
from networks.unet import UNet_CPS , UNet_MT ,UNet_UAMT  , UNet_EM , UNet_SLCNet , UNet_SSNet , UNet_SCP,UNet_PCCS,UNet_PCCS_ema
from networks.unet import ResUnet , ResDeepLabV3

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train" , prototype_num=4 , image_size=512 , patch_size = 4 , embed_dim = 48 , cls_head = True , contrast_embed=True , contrast_embed_dim=256 , contrast_embed_index=-3 , cfg = None):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == 'resunet':
        net = ResUnet(in_chns=in_chns, class_num=class_num,
                 encoder_name=cfg['network']['backbone'] ,
                 ).cuda()
    elif net_type == "resDeeplabV3":
        net = ResDeepLabV3(in_chns=in_chns, class_num=class_num,
                      encoder_name=cfg['network']['backbone'],
                      cfg=cfg).cuda()
    elif net_type == "unetsdf":
        net = UNet_sdf(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v1" and mode == "train":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "train":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "ecnet3d" and mode == "train":
        net = ECNet3d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "dicecenet3d" and mode == "train":
        net = DiceCENet3d(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v1" and mode == "test":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct" and mode == "train":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc" and mode == "train":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_scp" and mode == "train":
        net = UNet_SCP(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "resunet_scp" and mode == "train":
        from networks.resnet_unet_scp import ResUnet_SCP
        net = ResUnet_SCP(in_chns=in_chns, class_num=class_num,
                      encoder_name=cfg['network']['backbone'],
                      ).cuda()

    elif net_type == "unet_mt" and mode == "train":
        net = UNet_MT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_uamt" and mode == "train":
        net = UNet_UAMT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cps" and mode == "train":
        net = UNet_CPS(in_chns=in_chns, class_num=class_num).cuda()

    elif net_type == "unet_em" and mode == "train":
        net = UNet_EM(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_slc" and mode == "train":
        net = UNet_SLCNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ssnet" and mode == "train":
        net = UNet_SSNet(in_chns=in_chns, class_num=class_num).cuda()

    elif net_type == "unet_pccs" and mode == "train":
        prototype_num = cfg['protoseg']['num_prototype']
        net = UNet_PCCS(in_chns=in_chns, class_num=class_num,prototype_num=prototype_num).cuda()
    elif net_type == "unet_pccs_ema" and mode=="train":
        prototype_num = cfg['protoseg']['num_prototype']
        net = UNet_PCCS_ema(in_chns=in_chns, class_num=class_num , prototype_num=prototype_num).cuda()

    elif net_type == "std_unet_ugpcl":
        from networks.std_unet_ugpcl import StdUNetTF
        image_size = cfg['dataset']['kwargs']['resize_size'][0]
        patch_size = cfg['network']['download_patch']  # 标准UNET只能取2？？？
        embed_dim = cfg['network']['embed_dim']
        cls_head = cfg['network']['cls']
        contrast_embed = cfg['network']['contrast_embed']
        contrast_embed_dim = cfg['network']['contrast_embed_dim']
        contrast_embed_index = cfg['network']['contrast_embed_index']
        net = StdUNetTF(in_channels=in_chns,
                        classes=class_num,
                        img_size=image_size,
                        patch_size=patch_size,  # 标准UNET只能取2？？？
                        patches_resolution=[image_size // patch_size, image_size // patch_size],
                        window_size=8 if image_size == 256 else 7,
                        embed_dim=embed_dim , #普通unet只能设置这个，这个  数必须可以整除3，用来自注意力力，一分为3
                        contrast_embed=True).cuda()
    elif net_type == "resnet_ugpcl":
        from networks.resnet_unet_ugpcl import UNetTF
        image_size = cfg['dataset']['kwargs']['resize_size'][0]
        net = UNetTF(in_channels=in_chns,
                        classes=class_num,
                        img_size=image_size,
                     ).cuda()

    elif net_type == "unet_u2pl" and mode == "train":
        from networks.std_unet_u2pl import UNet_U2PL
        net = UNet_U2PL(in_chns=in_chns, class_num=class_num , cfg=cfg).cuda()
    elif net_type == 'unet_ucmt_ijcai' and mode == "train":
        from networks import unet_ucmt_ijcai  #里面的deeplabv3
        if cfg['network']['backbone'] == 'DeepLabv3p':
            net = unet_ucmt_ijcai.DeepLabv3p(in_channels=in_chns, out_channels=class_num).cuda()
        elif cfg['network']['backbone'] == 'UNet':
            net = unet_ucmt_ijcai.UNet_UCMT(in_channels=in_chns, out_channels=class_num).cuda()
        else:
            exit('not found backnone')
    elif net_type == 'DCT_2D' and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == 'ssm4mis_cnn_tf' and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()

    elif net_type == "bfdcnet2d" and mode == "train":
        net = BFDCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "dicecenetfuse" and mode == "train":
        net = DiceCENet3d_fuse(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "dicecenetfuse_2" and mode == "train":
        net = DiceCENet3d_fuse_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "dicecenetfuse2d":
        net = DiceCENet2d_fuse(in_chns=in_chns, class_num=class_num).cuda()
    return net
