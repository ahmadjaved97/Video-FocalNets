from timm.models import create_model
from . import videofocalnet
from . import uniformerv2_model #change

def build_model(config):
    model_type = config.MODEL.TYPE
    is_pretrained = config.MODEL.PRETRAINED 
    print(f"Creating model: {model_type}")
    
    if "focal" in model_type:
        model = create_model(
            model_type, 
            pretrained=is_pretrained, 
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            focal_levels=config.MODEL.FOCAL.FOCAL_LEVELS, 
            focal_windows=config.MODEL.FOCAL.FOCAL_WINDOWS, 
            use_conv_embed=config.MODEL.FOCAL.USE_CONV_EMBED, 
            use_layerscale=config.MODEL.FOCAL.USE_LAYERSCALE,
            use_postln=config.MODEL.FOCAL.USE_POSTLN, 
            use_postln_in_modulation=config.MODEL.FOCAL.USE_POSTLN_IN_MODULATION, 
            normalize_modulator=config.MODEL.FOCAL.NORMALIZE_MODULATOR,
            num_frames=config.DATA.NUM_FRAMES,
            tubelet_size=config.MODEL.TUBELET_SIZE
        )                      
    elif "vit" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif "resnet" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )
    
    elif model_type == "uniformerv2":
        print(f"Creating model: {model_type}")
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            checkpoint_num=config.MODEL.CHECKPOINT_NUM,
            num_classes=config.MODEL.NUM_CLASSES,
            t_size=config.DATA.NUM_FRAMES,
            
            backbone=config.UNIFORMERV2.BACKBONE,
            n_layers=config.UNIFORMERV2.N_LAYERS,
            n_dim=config.UNIFORMERV2.N_DIM,
            n_head=config.UNIFORMERV2.N_HEAD,
            mlp_factor=config.UNIFORMERV2.MLP_FACTOR,
            backbone_drop_path_rate=config.UNIFORMERV2.BACKBONE_DROP_PATH_RATE,
            drop_path_rate=config.UNIFORMERV2.DROP_PATH_RATE,
            mlp_dropout=config.UNIFORMERV2.MLP_DROPOUT,
            cls_dropout=config.UNIFORMERV2.CLS_DROPOUT,
            return_list=config.UNIFORMERV2.RETURN_LIST,

            temporal_downsample=config.UNIFORMERV2.TEMPORAL_DOWNSAMPLE,
            dw_reduction=config.UNIFORMERV2.DW_REDUCTION,
            no_lmhra=config.UNIFORMERV2.NO_LMHRA,
            double_lmhra=config.UNIFORMERV2.DOUBLE_LMHRA,

            frozen=config.UNIFORMERV2.FROZEN,
            delete_special_head=config.UNIFORMERV2.DELETE_SPECIAL_HEAD,
            pretrain=config.UNIFORMERV2.PRETRAINED_PATH,
        )
    else:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )        
    return model
