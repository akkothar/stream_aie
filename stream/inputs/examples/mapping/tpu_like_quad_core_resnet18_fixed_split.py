mapping = {
    '/avgpool/GlobalAveragePool': {'core_allocation': (4,)},
    '/conv1/Conv': {'core_allocation': (2,)},
    '/fc/Gemm': {'core_allocation': (2,)},
    '/layer1/layer1.0/Add': {'core_allocation': (5,)},
    '/layer1/layer1.0/conv1/Conv': {'core_allocation': (3,)},
    '/layer1/layer1.0/conv2/Conv': {'core_allocation': (3,)},
    '/layer1/layer1.1/Add': {'core_allocation': (5,)},
    '/layer1/layer1.1/conv1/Conv': {'core_allocation': (1,)},
    '/layer1/layer1.1/conv2/Conv': {'core_allocation': (1,)},
    '/layer2/layer2.0/Add': {'core_allocation': (5,)},
    '/layer2/layer2.0/conv1/Conv': {'core_allocation': (1,)},
    '/layer2/layer2.0/conv2/Conv': {'core_allocation': (1,)},
    '/layer2/layer2.0/downsample/downsample.0/Conv': {'core_allocation': (1,)},
    '/layer2/layer2.1/Add': {'core_allocation': (5,)},
    '/layer2/layer2.1/conv1/Conv': {'core_allocation': (1,)},
    '/layer2/layer2.1/conv2/Conv': {'core_allocation': (1,)},
    '/layer3/layer3.0/Add': {'core_allocation': (5,)},
    '/layer3/layer3.0/conv1/Conv': {'core_allocation': (1,)},
    '/layer3/layer3.0/conv2/Conv': {'core_allocation': (1,)},
    '/layer3/layer3.0/downsample/downsample.0/Conv': {'core_allocation': (1,)},
    '/layer3/layer3.1/Add': {'core_allocation': (5,)},
    '/layer3/layer3.1/conv1/Conv': {'core_allocation': (3,)},
    '/layer3/layer3.1/conv2/Conv': {'core_allocation': (1,)},
    '/layer4/layer4.0/Add': {'core_allocation': (5,)},
    '/layer4/layer4.0/conv1/Conv': {'core_allocation': (3, 1)},
    '/layer4/layer4.0/conv2/Conv': {'core_allocation': (3, 1, 2, 3)},
    '/layer4/layer4.0/downsample/downsample.0/Conv': {'core_allocation': (3,)},
    '/layer4/layer4.1/Add': {'core_allocation': (5,)},
    '/layer4/layer4.1/conv1/Conv': {'core_allocation': (0, 1, 3, 0)},
    '/layer4/layer4.1/conv2/Conv': {'core_allocation': (1, 0, 3, 1)},
    '/maxpool/MaxPool': {'core_allocation': (4,)}
}