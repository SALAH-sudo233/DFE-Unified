from .classifier import SpatialClassifierVN

def get_field_vn(config, num_classes, num_bond_types, in_sca, in_vec, ablation=None):
    if config.name == 'classifier':
        return SpatialClassifierVN(
            num_classes = num_classes,
            num_bond_types = num_bond_types,
            in_vec = in_vec,
            in_sca = in_sca,
            num_filters = [config.num_filters, config.num_filters_vec],
            edge_channels = config.edge_channels,
            num_heads = config.num_heads,
            k = config.knn,
            cutoff = config.cutoff,
            ablation = ablation,
        )
    else:
        raise NotImplementedError('Unknown field: %s' % config.name)
