from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.workload.computation_node import ComputationNode
from zigzag.utils import pickle_deepcopy
from zigzag.classes.cost_model.cost_model import get_total_inst_bandwidth

from stream.utils import get_too_large_operands


class FitnessEvaluator:
    def __init__(
        self, workload=None, accelerator=None, node_hw_performances=None
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError


class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload,
        accelerator,
        node_hw_performances,
        layer_groups_flexible,
        operands_to_prefetch,
        scheduling_order=None,
        results_path=None, # Aya
        memTile_flag=False, # Aya
        memTile_prefetch_flag=False, # Aya
        memTile_prefetch_count=4, # Aya
        memTile_eviction_flag=False, # Aya
        idle_num_for_mem_tile=2, # Aya
    ) -> None:
        super().__init__(workload, accelerator, node_hw_performances)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.layer_groups_flexible = layer_groups_flexible
        #self.scheduler_candidate_selection = scheduler_candidate_selection
        self.operands_to_prefetch = operands_to_prefetch
        #self.original_workload = original_workload
        #self.constant_operand_occupation_factor = 1
        #self.layer_stacks_mode = LayerStackMode.OCCUPATION_BASED
        self.scheduling_order = scheduling_order

        self.results_path = results_path

        self.memTile_flag = memTile_flag
        self.memTile_prefetch_flag = memTile_prefetch_flag 
        self.memTile_prefetch_count = memTile_prefetch_count
        self.memTile_eviction_flag = memTile_eviction_flag

        self.idle_num_for_mem_tile = idle_num_for_mem_tile

    def get_fitness(self, core_allocations: list, return_scme=False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        # layer_stacks = get_layer_stacks(
        #     self.workload,
        #     self.original_workload,
        #     self.accelerator,
        #     self.constant_operand_occupation_factor,
        #     self.layer_stacks_mode,
        # )


        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.operands_to_prefetch,
            self.scheduling_order,
            self.results_path, # Aya
            self.memTile_flag, # Aya
            self.memTile_prefetch_flag, # Aya
            self.memTile_prefetch_count, # Aya
            self.memTile_eviction_flag, # Aya
            self.idle_num_for_mem_tile,
        )

        # Aya: originally, this function used to not return anything, but now I made it return the 
        self.dbg_cns_start_end_cycles, self.dbg_tensors_transfer_details, self.full_dbg_transfer_prefetch_weights_end_timings, self.cores_prefetched_to = scme.run()

        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

     # Aya: added the following function to print some details about the produced schedule
    def print_to_file_cns_cycles_results(self, printing_file, cores):
        # first sort the list with respect to the start_cycle
        self.dbg_cns_start_end_cycles.sort(key=lambda x: x[3])
        #for core in cores:
            #print("\n", file=printing_file)
        for (from_offchip_flag, receiver_core_id, start_cycle, end_cycle, tensor_operand, tensor, CN) in self.dbg_cns_start_end_cycles:
            #if receiver_core_id == core.id:
            # if end_cycle == -1: 
            #     print("\tCore {} already has the tensor {} with operand {} in its L1 memory for {}.".format(receiver_core_id, tensor, tensor.layer_operand, CN), file=printing_file)
            # else:

            # if(from_offchip_flag == False):  
            #     if(tensor.layer_operand == "I" or tensor.layer_operand == "W"):
            #         print("Transfer of {} with operand {} to {} on Core {} starts at Cycle {}. It is consumed by {} and is NOT coming from the offchip core".format(tensor, tensor.layer_operand, CN, receiver_core_id, start_cycle, tensor.origin), file=printing_file)
            #     else:
            #         print("Transfer of {} with operand {} to {} on Core {} starts at Cycle {}. It is produced by {} and is NOT coming from the offchip core".format(tensor, tensor.layer_operand, CN, receiver_core_id, start_cycle, tensor.origin), file=printing_file)
            # else:
            #     if(tensor.layer_operand == "I" or tensor.layer_operand == "W"):
            #         print("Transfer of {} with operand {} to {} on Core {} starts at Cycle {}. It is consumed by {} and is coming from the offchip core".format(tensor, tensor.layer_operand, CN, receiver_core_id, start_cycle, tensor.origin), file=printing_file)
            #     else:
            #         print("Transfer of {} with operand {} to {} on Core {} starts at Cycle {}. It is produced by {} and is coming from the offchip core".format(tensor, tensor.layer_operand, CN, receiver_core_id, start_cycle, tensor.origin), file=printing_file)
            
            print("Start time of {} on Core {} is Cycle {}.".format(CN, receiver_core_id, start_cycle), file=printing_file)

    def print_to_file_tensors_transfers_end_cycles(self,  printing_file, cores):
        # first sort the list with respect to the end of the transfer
        self.dbg_tensors_transfer_details.sort(key=lambda x: x[3])
        #for core in cores:
            #print("\n", file=printing_file)
        for (from_offchip_flag, is_too_large_operands, receiver_core_id, transfer_complete_cycle, tensor_operand, tensor, CN) in self.dbg_tensors_transfer_details:
            if not is_too_large_operands:
                    #print("\tCore {} already has the tensor {} in its L1 memory for {}.".format(receiver_core_id, tensor, CN), file=printing_file)
                if transfer_complete_cycle != -1: 
                    if(from_offchip_flag == False):  
                        if(tensor.layer_operand == "I" or tensor.layer_operand == "W"):
                            print("NORMAL Transfer of {} to {} on Core {} is completed at Cycle {}. It is consumed by {} and is NOT coming from the offchip core".format(tensor, CN, receiver_core_id, transfer_complete_cycle, tensor.origin), file=printing_file)
                        else:
                            print("NORMAL Transfer of {} to {} on Core {} is completed at Cycle {}. It is produced by {} and is NOT coming from the offchip core".format(tensor, CN, receiver_core_id, transfer_complete_cycle, tensor.origin), file=printing_file)
                    else:
                        if(tensor.layer_operand == "I" or tensor.layer_operand == "W"):
                            print("NORMAL Transfer of {} to {} on Core {} is completed at Cycle {}. It is consumed by {} and is coming from the offchip core".format(tensor, CN, receiver_core_id, transfer_complete_cycle, tensor.origin), file=printing_file)
                        else:
                            print("NORMAL Transfer of {} to {} on Core {} is completed at Cycle {}. It is produced by {} and is coming from the offchip core".format(tensor, CN, receiver_core_id, transfer_complete_cycle, tensor.origin), file=printing_file)
            else:
                if tensor == ["I1"]:
                    actual_tensor_operand = "I"
                else:
                    if tensor == ["I2"]:
                        actual_tensor_operand = "W"
                    else:
                        actual_tensor_operand = "O"

                # if transfer_complete_cycle == -1: 
                    # print("\tCore {} already has the tensor {} in its L1 memory for {}.".format(receiver_core_id, actual_tensor_operand, CN), file=printing_file)
                if transfer_complete_cycle != -1:
                    if(from_offchip_flag == False):  
                        print("BLOCK Transfer of {} to {} on Core {} is completed at Cycle {}. It is NOT coming from the offchip core".format(actual_tensor_operand, CN, receiver_core_id, transfer_complete_cycle), file=printing_file)
                    else:
                        print("BLOCK Transfer of {} to {} on Core {} is completed at Cycle {}. It is coming from the offchip core".format(actual_tensor_operand, CN, receiver_core_id, transfer_complete_cycle), file=printing_file)
                        


#new_tensor.size, new_tensor.layer_operand, new_tensor.origin, new_tensor.loop_dimensions, new_tensor.loop_ranges

        # print("##### Inside transfer_tensor_to_core function of accelerator.py, printing the transfer_end #####", file=printing_file)
        # print("Sender core id is:{}".format(sender_core), file=printing_file)
        # print("Receiver core id is:{}".format(receiving_core_id), file=printing_file)
        # print("Transfer_start is:{} cycles".format(transfer_start), file=printing_file)
        # print("#######################################################################", file=printing_file)
    
    # Aya: added the following function to print 
    # Note that in some cases, we get no cores and 0 end time of prefetching
        # this probably indicates that all weights were initially contained inside the cores and there was no time needed for prefetching!
    def print_to_file_weights_prefetching_cycles_results(self, printing_file):
        print("One prefetch ends at Cycle {}".format(self.full_dbg_transfer_prefetch_weights_end_timings), file=printing_file)
        print("The cores that we prefectched weights to initially are: {}".format(self.cores_prefetched_to), file=printing_file)

    # Aya: added the following function to print the individual links used int the calculations for pairs of cores
    def print_to_file_used_links_between_cores(self, printing_file):
        for core_1 in self.accelerator.cores.nodes():
            for core_2 in self.accelerator.cores.nodes():
                if core_1 == core_2:
                    continue
                link = self.accelerator.communication_manager.pair_links[(core_1, core_2)]
                print("Sender Core: {}, Receiver Core: {} has Link: {}: ".format(core_1, core_2, link), file=printing_file)

    def set_node_core_allocations(self, core_allocations):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.nodes()
                if isinstance(node, ComputationNode)
                and node.id[0] == layer_id
                and node.group == group_id
            )
            for node in nodes:
                try:
                    equivalent_unique_node = next(
                        (n for n in self.node_hw_performances.keys() if node == n)
                    )
                except StopIteration:
                    raise ValueError(
                        f"The given node_hw_performances doesn't have run information for node={node}"
                    )
                try:
                    cme = self.node_hw_performances[equivalent_unique_node][core]
                except KeyError:
                    raise KeyError(
                        f"The given node_hw_performances doesn't have information for core_allocation={core_allocation} of node={node}"
                    )
                onchip_energy = (
                    cme.energy_total
                )  # Initialize on-chip energy as total energy
                latency = cme.latency_total1
                too_large_operands = get_too_large_operands(
                    cme, self.accelerator, core_id=core_allocation
                )
                # If there is a too_large_operand, we separate the off-chip energy.
                offchip_energy = 0
                for too_large_operand in too_large_operands:
                    layer_operand = next(
                        (
                            k
                            for (k, v) in cme.layer.memory_operand_links.items()
                            if v == too_large_operand
                        )
                    )
                    layer_operand_offchip_energy = cme.energy_breakdown[layer_operand][
                        -1
                    ]
                    offchip_energy += layer_operand_offchip_energy
                    onchip_energy -= layer_operand_offchip_energy
                # If there was offchip memory added for too_large_operands, get the offchip bandwidth
                offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
                offchip_instance = next(v for k, v in offchip_core.mem_hierarchy_dict.items())[-1].memory_instance
                offchip_bw = get_total_inst_bandwidth(cme, offchip_instance)
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(latency)
                node.set_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(offchip_bw)
