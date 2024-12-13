
class ACDCPath:
    def __init__(self , datasetname="acdc" ,num=10):
        self.config_pathsemi_unet = rf"configs/{datasetname}/percent_{num}/unet_config.yaml",
        self.config_pathfullunet = rf"configs/{datasetname}/full_supervised/unet_config.yaml",
        self.config_pathppcnet = rf"configs/{datasetname}/percent_{num}/ppcnt_config.yaml",
        self.config_pathpccs = rf"configs/{datasetname}/percent_{num}/pccs_config.yaml",

        self.config_pathugpcl = rf"configs/{datasetname}/percent_{num}/ugpcl_config.yaml",
        self.config_pathscpnet = rf"configs/{datasetname}/percent_{num}/scp_config.yaml",
        self.config_pathslcnet = rf"configs/{datasetname}/percent_{num}/slc_config.yaml",

        self.config_pathu2pl = rf"configs/{datasetname}/percent_{num}/u2pl_config.yaml",
        self.config_pathcps = rf"configs/{datasetname}/percent_{num}/cps_config.yaml",

        self.config_pathem = rf"configs/{datasetname}/percent_{num}/em_config.yaml",
        self.config_pathmt = rf"configs/{datasetname}/percent_{num}/mt_config.yaml",
        self.config_pathuamt = rf"configs/{datasetname}/percent_{num}/uamt_config.yaml",

        self.config_pathssnet = rf"configs/{datasetname}/percent_{num}/ssnet_config.yaml",
        self.config_pathmcnet = rf"configs/{datasetname}/percent_{num}/mcnet_config.yaml",
        self.config_pathcct = rf"configs/{datasetname}/percent_{num}/cct_config.yaml",
        self.config_pathurpc = rf"configs/{datasetname}/percent_{num}/urpc_config.yaml",
        self.config_pathucmt_ijcai = rf"configs/{datasetname}/percent_{num}/UCMT_ijcai_config.yaml",
        self.config_pathdct2d = rf"configs/{datasetname}/percent_{num}/DCT_2D_config.yaml",
        self.config_pathssm4mis = rf"configs/{datasetname}/percent_{num}/ssm4mis_config.yaml",

        self.config_pathbcp = rf"configs/{datasetname}/percent_{num}/bcp_config.yaml",

    def tolist(self):
        pass

class BUSIPath:  # BUSI_0_3_nocover
    def __init__(self , datasetname="BUSI_0_3_nocover" ,  num=10):
        self.config_pathsemi_unet = rf"configs/{datasetname}/percent_{num}/unet_config.yaml",
        self.config_pathfullunet = rf"configs/{datasetname}/full_supervised/unet_config.yaml",
        self.config_pathppcnet = rf"configs/{datasetname}/percent_{num}/ppcnt_config.yaml",
        self.config_pathpccs = rf"configs/{datasetname}/percent_{num}/pccs_config.yaml",

        self.config_pathugpcl = rf"configs/{datasetname}/percent_{num}/ugpcl_config.yaml",
        self.config_pathscpnet = rf"configs/{datasetname}/percent_{num}/scp_config.yaml",
        self.config_pathslcnet = rf"configs/{datasetname}/percent_{num}/slc_config.yaml",

        self.config_pathu2pl = rf"configs/{datasetname}/percent_{num}/u2pl_config.yaml",
        self.config_pathcps = rf"configs/{datasetname}/percent_{num}/cps_config.yaml",

        self.config_pathem = rf"configs/{datasetname}/percent_{num}/em_config.yaml",
        self.config_pathmt = rf"configs/{datasetname}/percent_{num}/mt_config.yaml",
        self.config_pathuamt = rf"configs/{datasetname}/percent_{num}/uamt_config.yaml",

        self.config_pathssnet = rf"configs/{datasetname}/percent_{num}/ssnet_config.yaml",
        self.config_pathmcnet = rf"configs/{datasetname}/percent_{num}/mcnet_config.yaml",
        self.config_pathcct = rf"configs/{datasetname}/percent_{num}/cct_config.yaml",
        self.config_pathurpc = rf"configs/{datasetname}/percent_{num}/urpc_config.yaml",

        self.config_pathucmt_ijcai = rf"configs/{datasetname}/percent_{num}/UCMT_ijcai_config.yaml",
        self.config_pathdct2d = rf"configs/{datasetname}/percent_{num}/DCT_2D_config.yaml",
        self.config_pathssm4mis = rf"configs/{datasetname}/percent_{num}/ssm4mis_config.yaml",
        self.config_pathbcp = rf"configs/{datasetname}/percent_{num}/bcp_config.yaml",


class BreastPath:
    def __init__(self , datasetname="breastUS1st_MRI2nd_all" ,  num=10):
        self.config_pathsemi_unet = rf"configs/{datasetname}/percent_{num}/unet_config.yaml",
        self.config_pathfullunet = rf"configs/{datasetname}/full_supervised/unet_config.yaml",
        self.config_pathppcnet = rf"configs/{datasetname}/percent_{num}/ppcnt_config.yaml",
        self.config_pathpccs = rf"configs/{datasetname}/percent_{num}/pccs_config.yaml",

        self.config_pathugpcl = rf"configs/{datasetname}/percent_{num}/ugpcl_config.yaml",
        self.config_pathscpnet = rf"configs/{datasetname}/percent_{num}/scp_config.yaml",
        self.config_pathslcnet = rf"configs/{datasetname}/percent_{num}/slc_config.yaml",

        self.config_pathu2pl = rf"configs/{datasetname}/percent_{num}/u2pl_config.yaml",
        self.config_pathcps = rf"configs/{datasetname}/percent_{num}/cps_config.yaml",

        self.config_pathem = rf"configs/{datasetname}/percent_{num}/em_config.yaml",
        self.config_pathmt = rf"configs/{datasetname}/percent_{num}/mt_config.yaml",
        self.config_pathuamt = rf"configs/{datasetname}/percent_{num}/uamt_config.yaml",

        self.config_pathssnet = rf"configs/{datasetname}/percent_{num}/ssnet_config.yaml",
        self.config_pathmcnet = rf"configs/{datasetname}/percent_{num}/mcnet_config.yaml",
        self.config_pathcct = rf"configs/{datasetname}/percent_{num}/cct_config.yaml",
        self.config_pathurpc = rf"configs/{datasetname}/percent_{num}/urpc_config.yaml",

        self.config_pathucmt_ijcai = rf"configs/{datasetname}/percent_{num}/UCMT_ijcai_config.yaml",
        self.config_pathdct2d = rf"configs/{datasetname}/percent_{num}/DCT_2D_config.yaml",
        self.config_pathssm4mis = rf"configs/{datasetname}/percent_{num}/ssm4mis_config.yaml",

        self.config_pathbcp = rf"configs/{datasetname}/percent_{num}/bcp_config.yaml",

def select_dataset_methods(nums=1):
    acdc_path = ACDCPath(datasetname='acdc' , num=nums)
    busi_path = BUSIPath(datasetname='BUSI_original' , num=nums)
    bmL2021_path = BreastPath(datasetname='BML_augmerge_1674' , num=nums)


    acdc_config_path_list = [
        # acdc_path.config_pathsemi_unet,
        # acdc_path.config_pathfullunet,
        # acdc_path.config_pathppcnet,
        acdc_path.config_pathpccs,
        # acdc_path.config_pathbcp,

        # acdc_path.config_pathugpcl,
        # acdc_path.config_pathscpnet,
        # acdc_path.config_pathslcnet,

        # acdc_path.config_pathu2pl,
        # acdc_path.config_pathcps,
        # acdc_path.config_pathem,
        # acdc_path.config_pathmt,
        # acdc_path.config_pathuamt,
        #
        # acdc_path.config_pathssnet,
        # acdc_path.config_pathmcnet,
        # acdc_path.config_pathcct,
        # acdc_path.config_pathurpc,
        # acdc_path.config_pathucmt_ijcai,
        # acdc_path.config_pathdct2d,
        # acdc_path.config_pathssm4mis,
    ]
    busi_config_path_list = [
        # busi_path.config_pathfullunet,
        # busi_path.config_pathsemi_unet,
        #
        busi_path.config_pathpccs,
        # busi_path.config_pathppcnet,
        busi_path.config_pathbcp,



        # busi_path.config_pathugpcl,
        # busi_path.config_pathscpnet,
        #  busi_path.config_pathslcnet,
        #
        # busi_path.config_pathu2pl,
        # busi_path.config_pathcps,
        # busi_path.config_pathem,
        # busi_path.config_pathmt,
        # busi_path.config_pathuamt,
        #
        # busi_path.config_pathssnet,
        # busi_path.config_pathmcnet,
        # busi_path.config_pathcct,
        # busi_path.config_pathurpc,
        # busi_path.config_pathucmt_ijcai,
        # busi_path.config_pathdct2d,
        # busi_path.config_pathssm4mis,
    ]


    bmL_merge_2021_config_path_list = [
        # 2021年标注
        # bmL2021_path.config_pathfullunet,
        # bmL2021_path.config_pathsemi_unet,

        # bmL2021_path.config_pathppcnet,
        bmL2021_path.config_pathpccs,
        bmL2021_path.config_pathbcp,
        # bmL2021_path.config_pathugpcl,
        bmL2021_path.config_pathscpnet,
        # bmL2021_path.config_pathslcnet,

        # bmL2021_path.config_pathu2pl,
        # bmL2021_path.config_pathcps,
        # bmL2021_path.config_pathem,
        # bmL2021_path.config_pathmt,
        # bmL2021_path.config_pathuamt,
        #
        # bmL2021_path.config_pathssnet,
        # bmL2021_path.config_pathmcnet,
        # bmL2021_path.config_pathcct,
        # bmL2021_path.config_pathurpc,

        # bmL2021_path.config_pathdct2d
        # bmL2021_path.config_pathssm4mis,
    ]




    print(acdc_config_path_list, busi_config_path_list,
          bmL_merge_2021_config_path_list   )

    return acdc_config_path_list
