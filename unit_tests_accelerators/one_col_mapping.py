# This file is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
 
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#
#===----------------------------------------------------------------------===//

mapping = {
    "default": {
        "core_allocation": [0, 1, 2, 3],
    },
    "Conv": {
        "core_allocation": [0, 1, 2, 3],
    },
    "Gemm": {
        "core_allocation": [0, 1, 2, 3],
    },
    "Pool": {
        "core_allocation": 5,
    },
    "MaxPool": {
        "core_allocation": 5,
    },
    "AveragePool": {
        "core_allocation": 5,
    },
    "GlobalAveragePool": {
        "core_allocation": 5,
    },
    "Add": {
        "core_allocation": 3,
    },
    "Identity": {
        "core_allocation": 3,
    },
}