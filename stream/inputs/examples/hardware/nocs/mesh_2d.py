import numpy as np
import networkx as nx
from networkx import DiGraph

from stream.classes.hardware.architecture.communication_link import CommunicationLink
from zigzag.classes.hardware.architecture.core import Core

# From the AIE-MLs perspective, the throughput of each of the loads and store is 256 bits per clock cycle.
aya_core_to_core_bw = 256  # bandwidth of every link connecting two neighboring cores
aya_core_to_mem_tile_bw = 32 * 6
aya_everything_to_dram_bw = 64 * 8


def have_shared_memory(a, b):
    """Returns True if core a and core b have a shared top level memory

    Args:
        a (Core): First core
        b (Core): Second core
    """
    top_level_memory_instances_a = set(
        [
            level.memory_instance
            for level, out_degree in a.memory_hierarchy.out_degree()
            if out_degree == 0
        ]
    )
    top_level_memory_instances_b = set(
        [
            level.memory_instance
            for level, out_degree in b.memory_hierarchy.out_degree()
            if out_degree == 0
        ]
    )
    for memory_instance_a in top_level_memory_instances_a:
        if memory_instance_a in top_level_memory_instances_b:
            return True
    return False


def get_2d_mesh(
    cores,
    nb_rows,
    nb_cols,
    bandwidth,
    unit_energy_cost,
    pooling_core=None,
    simd_core=None,
    offchip_core=None,
):
    """Return a 2D mesh graph of the cores where each core is connected to its N, E, S, W neighbour.
    We build the mesh by iterating through the row and then moving to the next column.
    Each connection between two cores includes two links, one in each direction, each with specified bandwidth.
    Thus there are a total of ((nb_cols-1)*2*nb_rows + (nb_rows-1)*2*nb_cols) links in the noc.
    If a pooling_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols more links are added.
    If an offchip_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols (+2 if a pooling core is present)

    Args:
        cores (list): list of core objects
        nb_rows (int): the number of rows in the 2D mesh
        nb_cols (int): the number of columns in the 2D mesh
        bandwidth (int): bandwidth of each created directional link in bits per clock cycle
        unit_energy_cost (float): The unit energy cost of having a communication-link active. This does not include the involved memory read/writes.
        pooling_core (Core, optional): If provided, the pooling core that is added.
        simd_core (Core, optional): If provided, the simd core that is added.
        offchip_core (Core, optional): If provided, the offchip core that is added.
        offchip_bandwidth (int, optional): If offchip_core is provided, this is the
    """

    cores_array = np.asarray(cores).reshape((nb_rows, nb_cols), order="F")

    edges = []
    # Horizontal edges
    for row in cores_array:
        # From left to right
        pairs = zip(row, row[1:])
        for pair in pairs:
            (sender, receiver) = pair
            if not have_shared_memory(sender, receiver):
                edges.append(
                    (
                        sender,
                        receiver,
                        {
                            "cl": CommunicationLink(
                                sender, receiver, aya_core_to_core_bw, unit_energy_cost
                            )
                        },
                    )
                )

        # From right to left
        pairs = zip(reversed(row), reversed(row[:-1]))
        for pair in pairs:
            (sender, receiver) = pair
            if not have_shared_memory(sender, receiver):
                edges.append(
                    (
                        sender,
                        receiver,
                        {
                            "cl": CommunicationLink(
                                sender, receiver, aya_core_to_core_bw, unit_energy_cost
                            )
                        },
                    )
                )
           

    # Vertical edges
    for col in cores_array.T:
        # From top to bottom (bottom is highest idx)
        pairs = zip(col, col[1:])
        for pair in pairs:
            (sender, receiver) = pair
            if not have_shared_memory(sender, receiver):
                edges.append(
                    (
                        sender,
                        receiver,
                        {
                            "cl": CommunicationLink(
                                sender, receiver, aya_core_to_core_bw, unit_energy_cost
                            )
                        },
                    )
                )
            
        # From bottom to top
        pairs = zip(reversed(col), reversed(col[:-1]))
        for pair in pairs:
            (sender, receiver) = pair
            if not have_shared_memory(sender, receiver):
                edges.append(
                    (
                        sender,
                        receiver,
                        {
                            "cl": CommunicationLink(
                                sender, receiver, aya_core_to_core_bw, unit_energy_cost
                            )
                        },
                    )
                )
           
    # If there is an offchip core, add a single link for writing to and a single link for reading from the offchip
    if offchip_core:
        # offchip_read_bandwidth = offchip_core.mem_r_bw_dict["O"][0]
        # offchip_write_bandwidth = offchip_core.mem_w_bw_dict["O"][0]

        offchip_bandwidth = aya_everything_to_dram_bw
        generic_test_link = CommunicationLink(
            "Any", "Any", offchip_bandwidth, unit_energy_cost
        )

        if not isinstance(offchip_core, Core):
            raise ValueError("The given offchip_core is not a Core object.")
        for core in cores:
            edges.append((core, offchip_core, {"cl": generic_test_link}))
            edges.append((offchip_core, core, {"cl": generic_test_link}))

        # for core_1 in cores:
        #     for core_2 in cores:
        #         if core_1 == core_2:
        #             continue
        #         edges.append((core_1, core_2, {"cl": generic_test_link}))   
        #         #edges.append((core_2, core_1, {"cl": generic_test_link}))  

    # Build the graph using the constructed list of edges
    H = DiGraph(edges)

    return H
