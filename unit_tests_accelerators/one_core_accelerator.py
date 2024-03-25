from unit_tests_accelerators.aie_core1_small_memory import (
    get_core as one_aie_core,
)
from stream.inputs.aie.hardware.shim_dma_core import (
    get_shim_dma_core as shim_core
)
from stream.inputs.examples.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

# changed all cores to be instances of aie_core1 since all AIE tiles should be identical
cores = [one_aie_core(0)] 
offchip_core_id = 6
aya_everything_to_dram_bw = 64 * 8

offchip_core = shim_core(id=offchip_core_id, offchip_bw=aya_everything_to_dram_bw) # basically DRAM

aya_everything_to_dram_bw = 64 * 8
offchip_core = shim_core(id=offchip_core_id, offchip_bw=aya_everything_to_dram_bw) # basically DRAM

parallel_links_flag = True # Aya: added this to selectively choose if the exploration includes multiple parallel links between a pair of cores or just the shortest links..

nb_rows= 1
nb_cols= 1
# Comment out the offchip_bandwidth because we can get this attribute from the offchip_core (if it is defined), thus no need to manually define it
cores_graph = get_2d_mesh(
    cores,
    parallel_links_flag,  # Aya: added this to selectively choose if the exploration includes multiple parallel links between a pair of cores or just the shortest links..
    nb_rows=nb_rows,
    nb_cols=nb_cols,
    axi_bandwidth=aya_everything_to_dram_bw,
    pooling_core=[],
    unit_energy_cost=0,
    offchip_read_channels_num=1, 
    offchip_write_channels_num=1, 
    memTile_read_channels_num=1,
    memTile_write_channels_num=1,
    offchip_core=offchip_core,
)  # , offchip_bandwidth=32)

accelerator = Accelerator(
    "AIE2_IPU", cores_graph, nb_rows, nb_cols, parallel_links_flag, offchip_core_id=offchip_core_id
)
