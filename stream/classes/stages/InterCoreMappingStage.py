from operator import attrgetter
import logging

from zigzag.classes.stages.Stage import Stage
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.opt.allocation.genetic_algorithm.genetic_algorithm import (
    GeneticAlgorithm,
)
from stream.classes.opt.allocation.genetic_algorithm.fitness_evaluator import (
    StandardFitnessEvaluator,
)
from stream.utils import get_too_large_operands
from zigzag.classes.cost_model.cost_model import get_total_inst_bandwidth

# Aya
from zigzag.visualization.results.print_mapping import (
    print_mapping as aya_print_mapping, 
    print_good_tm_format as aya_print_good_tm_format,
    print_printing_block as aya_print_printing_block,
)
# Aya:
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)


class InterCoreMappingStage(Stage):
    """
    Class that finds the best inter-core mapping using a genetic algorithm.
    From the IntraCoreMappingStage we receive the `node_hw_performances`, containing for each node and its valid core allocations the best CME.
    We then initialize the genetic algorithm.
    TODO A separate "GeneticAlgorithmStage" should be added where we parse all GA-related info and this stage then calls that stage.
    """

    def __init__(
        self,
        list_of_callables,
        *,
        workload,
        accelerator,
        node_hw_performances,
        nb_ga_generations,
        nb_ga_individuals,
        plot_hof,
        plot_file_name,
        plot_full_schedule=False,
        plot_data_transfer=False,
        operands_to_prefetch,
        **kwargs,
    ):
        """Initialize the InterCoreMappingStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            node_hw_performances (dict): A nested dict containing for each node a dict with for each valid core its best HW performance
            nb_ga_generations (int): The number of generations considered by the genetic algorithm
            nb_ga_individuals (int): The number of individuals in each genetic algorithm generation
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        self.nb_generations = nb_ga_generations
        self.nb_individuals = nb_ga_individuals
        self.plot_hof = plot_hof
        self.fig_path = plot_file_name
        self.plot_full_schedule = plot_full_schedule
        self.plot_data_transfer = plot_data_transfer
        self.operands_to_prefetch = operands_to_prefetch
        
        # Aya
        self.aya_dfg = kwargs["aya_dfg"]
        self.memTile_flag = kwargs["memTile_flag"]
        self.memTile_prefetch_flag = kwargs["memTile_prefetch_flag"]
        self.memTile_prefetch_count = kwargs["memTile_prefetch_count"]

        self.memTile_eviction_flag = kwargs["memTile_eviction_flag"]

        #self.original_workload = kwargs["original_workload"]
        self.scheduling_order = kwargs.get("scheduling_order", None)

        # Aya: added this to customize the path to the output
        self.results_path = kwargs["results_path"]

         # Aya: added this to print the original DFG workload before applying any splitting
        # with open(self.results_path + "/testing_cns_preds_Original_DFG_in_InterStage.txt", "a") as ff:
        #     self.print_cns_preds(ff)

        # Determine the set of all (layer, group) combinations to be allocated separately
        self.layer_groups = sorted(
            set((n.id[0], n.group) for n in self.workload.nodes())
        )

        # self.coarse_node_ids contains all the original node (aka layers) ids of the original graph
        self.unique_nodes = list(
            set((n for n, hw_performances in self.node_hw_performances.items()))
        )
        self.coarse_node_ids = [id[0] for id in self.layer_groups]
        # self.coarse_node_ids_flexible contains only those original node ids that have flexibility: they can be allocated to more than one core
        self.unique_nodes_flexible = sorted(
            set(
                (
                    n
                    for n, hw_performances in self.node_hw_performances.items()
                    if len(hw_performances.keys()) > 1
                )
            ),
            key=attrgetter("id"),
        )
        self.coarse_node_ids_flexible = [n.id[0] for n in self.unique_nodes_flexible]
        # For each unique node get the possible core allocations by getting the ids of the cores in node_hw_performances
        self.valid_allocations = []
        # Save all the layer group combinations that are flexible
        self.layer_groups_flexible = []
        for layer_id, group_id in self.layer_groups:
            # Find the unique node that corresponds to this layer
            # This assumes all the nodes of this layer are identical
            unique_node = next((n for n in self.unique_nodes if n.id[0] == layer_id))
            if unique_node in self.unique_nodes_flexible:
                hw_performances = self.node_hw_performances[unique_node]
                valid_core_ids = [core.id for core in hw_performances.keys()]
                self.layer_groups_flexible.append((layer_id, group_id))
                self.valid_allocations.append(valid_core_ids)

        # Set the hardware performance and core_allocation of nodes in the workload that only have a single possible core allocation
        self.set_hw_performance_non_flexible_nodes()

        # Initialize the fitness evaluator of different core allocations
        
        self.fitness_evaluator = StandardFitnessEvaluator(
            self.workload,
            self.accelerator,
            self.node_hw_performances,
            self.layer_groups_flexible,
            self.operands_to_prefetch,
            self.scheduling_order,
            self.results_path,
            self.memTile_flag,
            self.memTile_prefetch_flag,
            self.memTile_prefetch_count,
            self.memTile_eviction_flag,
        )

        # Extract the length of an individual.
        # This is the number of unique original nodes that have more than one possible core allocation
        self.individual_length = len(self.layer_groups_flexible)
        # Extract the value range each gene in the individual can have.
        # This ranges from 0 to the max core index.
        # TODO There might be some case where a core is not possible, so it shouldnt be tried by the GA
        core_ids = sorted([core.id for core in self.accelerator.cores.nodes()])
        self.core_id_range = (min(core_ids), max(core_ids))
        self.nb_cores = (
            max(core_ids) - min(core_ids) + 1
        )  # Assuming they are incrementing with step size 1

    # Aya: added this function to print information about the depth of object fifos by checking the number of predecessors for each CN in the workload graph
    def print_cns_preds(self, printing_file):
        if self.aya_dfg.__len__() == 0:
            print("There are no dependencies between the computation nodes!\n", file=printing_file)
        for node in self.aya_dfg:
            print("Predecessors of Node {} are: {}\n".format(node.id, list((self.aya_dfg.predecessors(node)))), file=printing_file)
        #print(nx.dfs_successors(self.workload), file=printing_file)

    def run(self):
        """Run the InterCoreMappingStage by checking if we have a fixed core_allocation.
        - if yes: evaluate fixed core allocation
        - if no: initialize and run the genetic algorithm
        """

        logger.info(f"Start InterCoreMappingStage.")

        # Aya: paths to files for exporting useful information about the scheduling and the mapping outputs
        cns_start_end_cycles_printing_file = self.results_path+"/check_cns_start_end_cycles.txt"
        tensors_transfer_end_cycles = self.results_path+"/check_tensors_end_transfer_cycles.txt"
        prefetch_weights_printing_file = self.results_path+"/check_weights_prefetch_transfer_cycles.txt"
        actual_links_printing_file = self.results_path+"/check_actual_cores_links.txt"
        mapping_output_printing_file = self.results_path+"/mapping_output.txt"

        # print("****************************************")
        # print(nx.dfs_successors(self.workload))
        # print("****************************************")

        if self.individual_length == 0:
            logger.info(f"Evaluating fixed layer-core allocation.")
            core_allocations = []
            (energy, latency, scme) = self.fitness_evaluator.get_fitness(
                core_allocations, return_scme=True
            )
            """
            scme.plot_schedule(plot_full_schedule=self.plot_full_schedule,
                               plot_data_transfer=self.plot_data_transfer,
                               fig_path=f"outputs/schedule_plot{self.fig_path}fixed.png")
            scme.plot_memory_usage(fig_path=f"outputs/memory_usage_plot{self.fig_path}fixed.png")
            """

            # Aya: this function prints to file the cycles at the beginning of tensor transfers between cores to help us understand the final schedule
            with open(cns_start_end_cycles_printing_file, "a") as ff:
                self.fitness_evaluator.print_to_file_cns_cycles_results(ff, self.accelerator.cores.nodes())

            with open(tensors_transfer_end_cycles, "a") as ff:
                self.fitness_evaluator.print_to_file_tensors_transfers_end_cycles(ff, self.accelerator.cores.nodes())

            with open(prefetch_weights_printing_file, "a") as ff:
                self.fitness_evaluator.print_to_file_weights_prefetching_cycles_results(ff)

            with open(actual_links_printing_file, "a") as ff:
                self.fitness_evaluator.print_to_file_used_links_between_cores(ff)

            with open(self.results_path+"/cns_preds_objectFifoDepth.txt", "a") as ff:
                self.print_cns_preds(ff)

            yield scme, None
        else:
            logger.info(
                f"Running Inter-Core Allocation Optimization with Genetic Algorithm."
            )

            # Aya: added the following to plot a graph of the explored design space
            explored_energies = []
            explored_latencies = []

            # Initialize the genetic algorithm
            self.genetic_algorithm = GeneticAlgorithm(
                self.fitness_evaluator,
                self.individual_length,
                self.valid_allocations,
                self.nb_generations,
                self.nb_individuals,
            )
            # Run the genetic algorithm and get the results
            pop, hof = self.genetic_algorithm.run()
            logger.info(f"Finished Genetic Algorithm.")
            print("Hall of fame:")
            print(hof)
            if self.plot_hof:
                for i, core_allocations in enumerate(hof):
                    # results = self.fitness_evaluator.get_fitness(
                    #     core_allocations, return_scme=True
                    # )
                    #scme = results[-1]
                    (energy, latency, scme) = self.fitness_evaluator.get_fitness(
                        core_allocations, return_scme=True
                    )
                    explored_energies.append(energy)
                    explored_latencies.append(latency)


                    save_last_core_allocation = core_allocations  # Aya added this

                    """
                    scme.plot_schedule(plot_full_schedule=self.plot_full_schedule,
                                       plot_data_transfer=self.plot_data_transfer,
                                       fig_path=f"outputs/schedule_plot{self.fig_path}{i}.png")
                    scme.plot_memory_usage(fig_path=f"outputs/memory_usage_plot{self.fig_path}{i}.png")
                    """
                
                # plt.scatter(explored_latencies, explored_energies, c ="blue")
                # plt.show()
                # plt.savefig('design_space.png')
                    
                # Aya: these functions print to files the cycles at the beginning of tensor transfers between cores to help us understand the final schedule
                # I'm printing it after the loop to use the final scme and the final transfer cycles
                with open(cns_start_end_cycles_printing_file, "a") as ff:
                    self.fitness_evaluator.print_to_file_cns_cycles_results(ff, self.accelerator.cores.nodes())

                with open(tensors_transfer_end_cycles, "a") as ff:
                    self.fitness_evaluator.print_to_file_tensors_transfers_end_cycles(ff, self.accelerator.cores.nodes())

                with open(prefetch_weights_printing_file, "a") as ff:
                    self.fitness_evaluator.print_to_file_weights_prefetching_cycles_results(ff)

                with open(actual_links_printing_file, "a") as ff:
                    self.fitness_evaluator.print_to_file_used_links_between_cores(ff)

                with open(self.results_path+"/cns_preds_objectFifoDepth.txt", "a") as ff:
                    self.print_cns_preds(ff)

                ############## Aya: added the following code to extract the final cmes after the genetic algorithm and fitness_evaluator
                old_layer_id = -1
                for i, core_allocation in enumerate(save_last_core_allocation):
                    core = self.accelerator.get_core(core_allocation)

                    (layer_id, group_id) = self.layer_groups_flexible[i]
                    # the mapping of all CNs of one layer should be identical so print only a single CN of each layer
                    if(layer_id == old_layer_id):   
                        continue
                    old_layer_id = layer_id

                    nodes = (
                        node
                        for node in self.workload.nodes()
                        if isinstance(node, ComputationNode)
                        and node.id[0] == layer_id
                        and node.group == group_id
                    )

                    old_cme = []
                    for node in nodes:
                        equivalent_unique_node = next(
                            (n for n in self.node_hw_performances.keys() if node == n)
                        )
                        cme = self.node_hw_performances[equivalent_unique_node][core]

                        if(cme == old_cme):
                            continue
                        old_cme = cme

                        with open(mapping_output_printing_file, "a") as ff:
                            print("############ The mapping results for one layer are ############", file=ff)
                            print("Number of elements at each memory level is {}".format(cme.mapping.data_elem_per_level), file=ff) # Zigzag's summary of tile sizes
                            print("Number of bits at each memory level is {}\n".format(cme.mapping.data_bit_per_level), file=ff) # Zigzag's summary of tile sizes
                            # print("Number of elements unrolled at each memory level is {}".format(cme.mapping.data_elem_per_level_unrolled), file=ff) # Zigzag's summary of tile sizes
                            # print("Total Number of bits unrolled at each memory level is {}\n".format(cme.mapping.data_bit_per_level_unrolled), file=ff) # Zigzag's summary of tile sizes
                            print("Spatial mapping field of the CME: {}".format(cme.spatial_mapping), file=ff)
                            # aya_print_mapping(cme, core, ff, self.memTile_flag)  # prints the table detailing the mapping
                            print("############ End of the mapping results of one layer ############", file=ff)
            yield scme, None
        logger.info(f"Finished InterCoreMappingStage.")

    def set_hw_performance_non_flexible_nodes(self):
        """Set the energy, runtime and core_allocation of the nodes in self.workload that only have a single possible core allocation."""
        non_flexible_unique_nodes = set(self.unique_nodes) - set(
            self.unique_nodes_flexible
        )
        for non_flexible_unique_node in non_flexible_unique_nodes:
            hw_performances = self.node_hw_performances[non_flexible_unique_node]
            assert (
                len(hw_performances.keys()) == 1
            ), f"Non-flexible unique node {non_flexible_unique_node} has more than one entry in node_hw_performances."
            (core, cme) = next((key, val) for key, val in hw_performances.items())
            onchip_energy = (
                cme.energy_total
            )  # Initialize the on-chip energy as total energy
            latency = cme.latency_total1
            core_allocation = core.id

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
                layer_operand_offchip_energy = cme.energy_breakdown[layer_operand][-1]
                offchip_energy += layer_operand_offchip_energy
                onchip_energy -= layer_operand_offchip_energy

            # If there was offchip memory added for too_large_operands, get the offchip bandwidth
            offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
            offchip_instance = next(v for k, v in offchip_core.mem_hierarchy_dict.items())[-1].memory_instance
            offchip_bw = get_total_inst_bandwidth(cme, offchip_instance)

            nodes = (n for n in self.workload.nodes() if n == non_flexible_unique_node and n.group == non_flexible_unique_node.group)
            for node in nodes:
                self.set_hw_performance_node(
                    node, onchip_energy, offchip_energy, latency, core_allocation
                )
                node.set_too_large_operands(too_large_operands.copy())
                node.set_offchip_bandwidth(offchip_bw)


    @staticmethod
    def set_hw_performance_node(
        node: ComputationNode,
        onchip_energy: float,
        offchip_energy: float,
        runtime: int,
        core_allocation: int,
    ):
        """Set the hardware performance and core_allocation of the given node.

        Args:
            node (Node): The node of which to set the
            onchip_energy (float): on-chip energy of executing this node
            offchip_energy (float): off-chip energy of executing this node
            runtime (int): runtime of executing this node
            core_allocation (int): the core_id on which this node will be ran
        """
        node.set_onchip_energy(onchip_energy)
        node.set_offchip_energy(offchip_energy)
        node.set_runtime(runtime)
        node.set_core_allocation(core_allocation)

    def is_leaf(self) -> bool:
        return True
