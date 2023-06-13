def build_backbone(backbone_name, feature_dim, num_input_c=3, mode='main'):
    if backbone_name == 'hourglass_gn':
        from rpin.models.backbones.hg_gn import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=num_input_c)
    elif backbone_name == 'hourglass_bn':
        from rpin.models.backbones.hg_bn import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=num_input_c, mode=mode)
    elif backbone_name == 'hourglass_in':
        from rpin.models.backbones.hg_in import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=num_input_c)
    elif backbone_name == 'hourglass_ln':
        from rpin.models.backbones.hg_ln import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=num_input_c)
    else:
        raise NotImplementedError
    return backbone
