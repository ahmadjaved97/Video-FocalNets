#!/usr/bin/env python
import json
import torch
import torch.nn as nn
from timm.models.registry import register_model

import uniformerv2_model as model
from .build import MODEL_REGISTRY

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class Uniformerv2(nn.Module):
    def __init__(self,
        use_checkpoint=False,
        checkpoint_num=[0],
        t_size=16, 
        dw_reduction=1.5,
        backbone_drop_path_rate=0., 
        temporal_downsample=True,
        no_lmhra=False, double_lmhra=True,
        return_list=[8, 9, 10, 11], 
        n_layers=4, 
        n_dim=768, 
        n_head=12, 
        mlp_factor=4.0, 
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, 
        num_classes=400,
        frozen=False,

        delete_special_head=False,
        pretrain='',


    ):
        super().__init__()

        # self.cfg = cfg

        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.num_classes = num_classes
        self.t_size = t_size

        self.backbone = backbone
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_head = n_head
        self.mlp_factor = mlp_factor
        self.backbone_drop_path_rate = backbone_drop_path_rate
        self.drop_path_rate = drop_path_rate
        self.mlp_dropout = mlp_dropout
        self.cls_dropout = cls_dropout
        self.return_list = return_list

        self.temporal_downsample = temporal_downsample
        self.dw_reduction = dw_reduction
        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra

        self.frozen = frozen

        self.delete_special_head = delete_special_head
        self.pretrain = pretrain

        # pre-trained from CLIP
        self.backbone = model.__dict__[backbone](
            use_checkpoint=self.use_checkpoint,
            checkpoint_num=self.checkpoint_num,
            t_size=self.t_size,
            dw_reduction=self.dw_reduction,
            backbone_drop_path_rate=self.backbone_drop_path_rate, 
            temporal_downsample=self.temporal_downsample,
            no_lmhra=self.no_lmhra,
            double_lmhra=self.double_lmhra,
            return_list=self.return_list, 
            n_layers=self.n_layers, 
            n_dim=self.n_dim, 
            n_head=self.n_head, 
            mlp_factor=self.mlp_factor, 
            drop_path_rate=self.drop_path_rate, 
            mlp_dropout=self.mlp_dropout, 
            cls_dropout=self.cls_dropout, 
            num_classes=self.num_classes,
            frozen=self.frozen,
        )

        if self.pretrain != '':
            # Load Kineti-700 pretrained model
            logger.info(f'load model from {self.pretrain}')
            state_dict = torch.load(self.pretrain, map_location='cpu')
            if self.delete_special_head and state_dict['backbone.transformer.proj.2.weight'].shape[0] != num_classes:
                logger.info('Delete FC')
                del state_dict['backbone.transformer.proj.2.weight']
                del state_dict['backbone.transformer.proj.2.bias']
            elif not self.delete_special_head:
                logger.info('Load FC')
                if num_classes == 400 or state_dict['backbone.transformer.proj.2.weight'].shape[0] == num_classes:
                    state_dict['backbone.transformer.proj.2.weight'] = state_dict['backbone.transformer.proj.2.weight'][:num_classes]
                    state_dict['backbone.transformer.proj.2.bias'] = state_dict['backbone.transformer.proj.2.bias'][:num_classes]
                else:
                    map_path = f'./data_list/k710/label_mixto{num_classes}.json'
                    logger.info(f'Load label map from {map_path}')
                    with open(map_path) as f:
                        label_map = json.load(f)
                    state_dict['backbone.transformer.proj.2.weight'] = state_dict['backbone.transformer.proj.2.weight'][label_map]
                    state_dict['backbone.transformer.proj.2.bias'] = state_dict['backbone.transformer.proj.2.bias'][label_map]
            self.load_state_dict(state_dict, strict=False)

        if frozen:
            backbone_list = [
                # Backbone
                'conv1', 'class_embedding', 'positional_embedding', 'ln_pre', 'transformer.resblocks'
            ]
            logger.info(f'Freeze List: {backbone_list}')
            for name, p in self.backbone.named_parameters():
                flag = False
                for module in backbone_list:
                    if module in name:
                        flag = True
                        break
                if flag:
                    logger.info(f'Frozen {name}')
                    p.requires_grad = False
                else:
                    logger.info(f'FT {name}')

    def forward(self, x):
        x = x[0]
        output = self.backbone(x)

        return output

@register_model
def uniformerv2_b16(
    pretrained=True, use_checkpoint=False, checkpoint_num=[0],
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[8, 9, 10, 11], 
    n_layers=4, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400,
    frozen=False,
    delete_special_head=False,
    pretrain='',
):
    model = Uniformerv2(
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction,
        backbone_drop_path_rate=backbone_drop_path_rate,
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list,
        n_layers=n_layers,
        n_dim=n_dim,
        n_head=n_head,
        mlp_factor=mlp_factor,
        drop_path_rate=drop_path_rate,
        mlp_dropout=mlp_dropout,
        cls_dropout=cls_dropout,
        num_classes=num_classes,
        frozen=frozen,
        delete_special_head=delete_special_head,
        pretrain=pretrain,
    )

    if pretrained and pretrain:
        logger.info(f"Loading pretrained weights from {pretrain}")
        state_dict = torch.load(pretrain, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    return model