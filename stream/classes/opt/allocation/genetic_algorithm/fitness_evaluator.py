from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.opt.scheduling.layer_stacks import get_layer_stacks, LayerStackMode
from stream.classes.workload.computation_node import ComputationNode
from zigzag.utils import pickle_deepcopy

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
        scheduler_candidate_selection,
        operands_to_prefetch,
        original_workload,  # used for layer stack calculation
        results_path, # Aya
    ) -> None:
        super().__init__(workload, accelerator, node_hw_performances)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.layer_groups_flexible = layer_groups_flexible
        self.scheduler_candidate_selection = scheduler_candidate_selection
        self.operands_to_prefetch = operands_to_prefetch
        self.original_workload = original_workload
        self.constant_operand_occupation_factor = 1
        self.layer_stacks_mode = LayerStackMode.OCCUPATION_BASED

        self.results_path = results_path

    def get_fitness(self, core_allocations: list, return_scme=False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        layer_stacks = get_layer_stacks(
            self.workload,
            self.original_workload,
            self.accelerator,
            self.constant_operand_occupation_factor,
            self.layer_stacks_mode,
        )
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.scheduler_candidate_selection,
            self.operands_to_prefetch,
            layer_stacks,
            self.results_path, # Aya
        )


        # Aya: originally, this function used to not return anything, but now I made it return the 
        self.full_dbg_transfer_timings, self.full_dbg_transfer_prefetch_weights_end_timings, self.cores_prefetched_to = scme.run()

        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

     # Aya: added the following function to print some details about the produced schedule
    def print_to_file_cycles_results(self, printing_file, cores):
        # first sort the list with respect to the start_cycle
        self.full_dbg_transfer_timings.sort(key=lambda x: x[2])
        #for core in cores:
            #print("\n", file=printing_file)
        for (from_offchip_flag, receiver_core_id, start_cycle, end_cycle, tensor_operand, tensor, CN) in self.full_dbg_transfer_timings:
            #if receiver_core_id == core.id:
            if end_cycle == -1: 
                print("\tCore {} already has the tensor of operand {} in its L1 memory for {}.".format(receiver_core_id, tensor.layer_operand, CN), file=printing_file)
            else:
                if(from_offchip_flag == False):  
                    print("Transfer of tensor operand {} to {} on Core {} starts at Cycle {} with origin {} and is NOT coming from the offchip core".format(tensor.layer_operand, CN, receiver_core_id, start_cycle, tensor.origin), file=printing_file)
                else:
                    print("Transfer of tensor operand {} to {} on Core {} starts at Cycle {} with origin {} and is coming from the offchip core".format(tensor.layer_operand, CN, receiver_core_id, start_cycle, tensor.origin), file=printing_file)

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
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(latency)
                node.set_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
