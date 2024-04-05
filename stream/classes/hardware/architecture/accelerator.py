from math import ceil
from networkx import DiGraph, MultiDiGraph

from zigzag.classes.hardware.architecture.core import Core
from stream.classes.cost_model.memory_manager import MemoryManager
from stream.classes.cost_model.communication_manager import CommunicationManager
from stream.classes.workload.tensor import Tensor

import numpy as np

class Accelerator:
    """
    The Accelerator class houses a set of Cores with an additional Global Buffer.
    This Global Buffer sits above the cores, and can optionally be disabled.
    In this Stream version, the cores are actually a graph with directed edges representing communication links.
    """

    def __init__(
        self,
        name,
        cores,  # Aya: this could be a Digraph or MultiDigraph depending on the parallel_links_flag 
        nb_rows, 
        nb_cols, 
        parallel_links_flag, # Aya: added this to selectively choose if the exploration includes multiple parallel links between a pair of cores or just the shortest links..
        offchip_core_id=None,
    ):
        self.name = name
        self.cores = cores
        self.offchip_core_id = offchip_core_id
        self.memory_manager = MemoryManager(self)
        self.parallel_links_flag = parallel_links_flag # Aya: added this to selectively choose if the exploration includes multiple parallel links between a pair of cores or just the shortest links..
        self.communication_manager = CommunicationManager(self)

        # Aya: added those extra fields to identify the memTile core in the same column as another compute core. This is used in the add_offchip_to_core function inside the IntraCoreMappingStage and should be ultimately used in the exploration of transfer_tensors
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, "cores": self.cores}

    def get_core(self, core_id: int or str) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        core = next((core for core in self.cores.nodes() if core.id == core_id), None)
        if not core:
            raise ValueError(
                f"Requested core with id {core_id} is not present in accelerator."
            )
        return core

    def spawn(
        self,
        tensor: Tensor,
        core: Core,
        memory_op: str,
        initial_timestep: int,
        available_timestep: int,
    ):
        """Spawns a tensor on a core.

        Args:
            tensor (Tensor): The tensor to be spawned.
            core (Core): The core on which to spawn the tensor.
            memory_op (str): The memory operand on the core where the tensor will spawn.
            initial_timestep (int): The timestep at which space will be reserved for the tensor.
            available_timestep (int): The timestep at which the tensor will become available. Different from initial_timestep when it is transferred.
        """
        self.memory_manager.add_tensor_to_core(
            tensor, core, initial_timestep, available_timestep, memory_op
        )

    def remove(self, tensor, core, memory_op, timestep, links_printing_file, dbg_memTile_file, write_back_to_offchip=False):
        """Remove tensor from core. If required, transfer to offchip before removal.

        Args:
            tensor (Tensor): The tensor to remove.
            core (Core): The Core to remove the tensor from.
            memory_op (str): The memory operand of the tensor.
            timestep (int): The timestep to remove the tensor at.
            write_back_to_offchip (bool, optional): Write the tensor to offchip before removal. Defaults to False.
        """

        ################################# STEP 1 #################################
        # Transfer the tensor to off-chip if required and not present there
        link_energy_cost = 0
        memory_energy_cost = 0
        memTile_core = [] # Aya
        offchip_instance = self.get_top_instance_of_core(
            self.offchip_core_id, memory_op
        )
        should_be_written_to_offchip = (
            write_back_to_offchip and not self.contains_tensor(tensor, offchip_instance)
        )
        current_timestep = timestep
        if should_be_written_to_offchip:
            (
                transfer_end,
                transfer_link_energy_cost,
                transfer_memory_energy_cost,
                eviction_link_energy_cost,
                eviction_memory_energy_cost,
                came_from_offchip,
                use_memTile_flag_status,
                memTile_core,
            ) = self.transfer_tensor_to_core(
                tensor,
                self.offchip_core_id,
                memory_op,
                non_evictable_tensors=[],
                memTile_flag=False,
                future_tensors=[], # Aya
                future_tensors_operands=[], # Aya
                all_future_tensors=[], # Aya
                links_printing_file=links_printing_file,
                dbg_memTile_file=dbg_memTile_file,
                sending_core_id=core.id,
            )
            # There should be no evictions as we are writing to offchip
            assert eviction_link_energy_cost == 0
            assert eviction_memory_energy_cost == 0
            assert not came_from_offchip
            link_energy_cost = transfer_link_energy_cost
            memory_energy_cost = transfer_memory_energy_cost
            current_timestep = max(current_timestep, transfer_end)

        ################################# STEP 2 #################################
        # Remove the tensor from the memory manager's attributes
        top_instance = self.get_top_instance_of_core(core, memory_op)
        self.memory_manager.remove_tensor_from_top_instance(
            top_instance,
            tensor,
            timestep,
        )

        return current_timestep, link_energy_cost, memory_energy_cost

    def remove_all(
        self, core, memory_operand, timestep, links_printing_file, dbg_memTile_file, exceptions=[], write_back_to_offchip=False
    ):
        """Remove all tensors from a core's memory with the given memory operand.
        If required, the tensors are written back to offchip before removal.

        Args:
            core (Core): The Core to remove the tensor from
            memory_operand (str): The memory operand for which all tensors should be evicted.
            timestep (int): The timestep to remove the tensor at.
            exceptions (list): A list of tensors that should not be evicted.
            write_back_to_offchip (bool, optional): Write the tensor to offchip before removal. Defaults to False.
        """
        total_link_energy_cost = 0
        total_memory_energy_cost = 0
        top_instance = self.get_top_instance_of_core(core, memory_operand)
        # stored_tensors = self.stored_tensors[core][top_level_idx]
        t = timestep
        for tensor in self.memory_manager.get_tensors_stored_at_timestep(
            top_instance, timestep
        ):
            if not tensor in exceptions:
                t, link_energy_cost, memory_energy_cost = self.remove(
                    tensor, core, memory_operand, t, links_printing_file, dbg_memTile_file, write_back_to_offchip
                )
                total_link_energy_cost += link_energy_cost
                total_memory_energy_cost += memory_energy_cost
        return t, total_link_energy_cost, total_memory_energy_cost

    def make_space_for(
        self,
        tensor: Tensor,
        core: Core,
        memory_op: str,
        timestep: int,
        links_printing_file: str,
        dbg_memTile_file: str,
        tensors_to_avoid_evicting: list = [],
    ):
        """Make space for the given tensor on the given core by evicting already stored tensors if necessary.

        Args:
            tensor (Tensor): The tensor to make space for.
            core (Core): The core where the tensor will be stored.
            memory_operand (str): The memory operand on the core.
            timestep (int): The timestep at which to make space for.
        """
        total_eviction_link_energy_cost = 0
        total_eviction_memory_energy_cost = 0

        top_instance = self.get_top_instance_of_core(core, memory_op)

        # Get the timestep at which there's enough space for this tensor
        enough_space_timestep = self.memory_manager.get_timestep_for_tensor_addition(
            tensor,
            core.id,
            timestep,
            memory_op=tensor.memory_operand,
        )

        tensors_to_evict = (
            self.memory_manager.find_best_tensor_combination_to_evict_fast(
                top_instance,
                tensor,
                enough_space_timestep,
                exceptions=tensors_to_avoid_evicting,
            )
        )
        if core.id == self.offchip_core_id and tensors_to_evict:
            raise ValueError(
                "Evictions required in offchip memory. Consider making offchip larger."
            )
        t_evictions_complete = timestep
        for tensor_to_evict in tensors_to_evict:
            (
                t_eviction_complete,
                eviction_link_energy_cost,
                eviction_memory_energy_cost,
            ) = self.remove(
                tensor_to_evict,
                core,
                memory_op,
                timestep,
                links_printing_file,
                dbg_memTile_file,
                write_back_to_offchip=True,
            )
            t_evictions_complete = max(t_evictions_complete, t_eviction_complete)
            total_eviction_link_energy_cost += eviction_link_energy_cost
            total_eviction_memory_energy_cost += eviction_memory_energy_cost
        t_evictions_complete = max(enough_space_timestep, t_evictions_complete)
        return (
            t_evictions_complete,
            total_eviction_link_energy_cost,
            total_eviction_memory_energy_cost,
        )
    
    # Aya: printing the link used in each tensor transfer by calling the following function
    def print_to_file_chosen_links_between_cores(self, sender_core, receiving_core, chosen_link, printing_file): 
        #print("The chosen link to transfer from {} to {} is {}".format(sender_core, receiving_core, chosen_link), file=printing_file)
        print("The link {} has active_ts = {}, active_deltas = {}, and tensors = {}".format(chosen_link, chosen_link.active_ts, chosen_link.active_deltas, chosen_link.tensors), file=printing_file)

    
    # Aya: use_memTile_flag_status can be one of three values: 
                # 0: indicating that we have enough offchip channels 
                # 1: indicating that the data is already present in the memTile and we should directly tansfer it from there
                # 2: indicating that the data is not present in the memTile and there is only one available offchip channel, so do offchip -> memTile then memTile -> core 
    def memTile_heuristic(self, memTile_core, use_heur_flag, tensor, tensor_operand, sender_core, receiving_core, evictions_complete_timestep):
        if not use_heur_flag:
            return 0
        # Two things:
        # (1) If the data is available inside the memTile, return memTile direct (because this is different from the offchip -> memTile -> Core)
        memTile_top_instance = self.get_top_instance_of_core(
            memTile_core.id, tensor_operand
        )
        
        if self.memory_manager.contains(tensor, memTile_top_instance):
            return 1

        links = self.communication_manager.get_links_for_pair(  
            sender_core, receiving_core
        )
        links_nested = []
        for path in links:
            if hasattr(path, '__iter__'):
                links_nested.append({link: link.bandwidth for link in path})  # Aya: to support multiple parallel paths between a pair of cores where each path could be made of multiple links (i.e., if the pair of cores are not directly connected)
            else:
                links_nested.append({path: path.bandwidth})
        # links = {link: link.bandwidth for link in links} # added for broadcasting
        links = links_nested

        transfer_start, transfer_duration, chosen_links, all_links_transfer_start_end = self.communication_manager.get_links_idle_window(
            links, evictions_complete_timestep, [tensor,], sender_core, receiving_core
        )

        ################## dbg prints
        # with open("check_memTile_condition.txt", "a") as ff:
        #     print("\t The earliest start time is {} and the links start times are:".format(evictions_complete_timestep), file=ff)
        #     print("Printing all_links: {}".format(all_links_transfer_start_end), file=ff)
        #     for link in all_links_transfer_start_end:
        #         for s, e, broadcast_flag in link:
        #             print("one link start time is {}".format(s), file=ff)
        ###########################################################

        # loop over the start and the end of all of the links connecting the sender (offchip) and receiver, and count the ones of them that have s = evictions_complete_timestep (because this indicates that the link is idle and ready to transfer immediately)
        idle_count = 0
        for link in all_links_transfer_start_end:
            for s, e, broadcast_flag in link:
                if s == evictions_complete_timestep:
                    idle_count += 1
        if idle_count < 2: # this means either there are no available links at all or there is only one link left
            return 2 
        return 0
        # (2) If the data is not available inside the memTile, check if only one link of the offchip is idle and all the rest are busy, return 2; otherwise, return 0
    
    def transfer_tensor_to_core(
        self,
        tensor: Tensor,
        receiving_core_id: int,
        tensor_operand: str,
        non_evictable_tensors: list,
        memTile_flag: bool, # Aya
        future_tensors: list, # Aya
        future_tensors_operands: list, # Aya
        all_future_tensors: list, # Aya: needed in case of necessary evictions
        links_printing_file: str,  # Aya
        dbg_memTile_file: str, # Aya
        sending_core_id: int = None,
    ):
        """
        Transfer a tensor to a given core id.
        If the tensor is already present on the receiving core, nothing happens.

        This function computes when the transfer can take place based on three factors:
        1) The tensor is available for transfer on a sender core.
        2) The receiver core has enough space to store the tensor.
        3) The links between sender and receiver have a long enough idle window.

        TODO: The transfer is scheduled as close as possible to the computation

        The tensor is then added to the memory. Evictions are still possible if
        there wasn't enough space on the receiver core at any earlier timestep.

        Args:
            tensor (Tensor): The tensor to transfer.
            receiving_core_id (int): The id of the core that needs to receive the tensor.
            tensor_operand (str): The memory operand where the tensor needs to be stored.
            non_evictable_tensors (list): the stored tensor that cannot be evicted
            sending_core_id (int, optional): The id of the core that should transfer the tensor.
        """
        ################################# STEP 0 #################################
        # Check if the tensor is already on the receiving core
        # Get the top instance where the tensor will be transferred to
        receiving_core = self.get_core(receiving_core_id)
        receiving_top_instance = self.get_top_instance_of_core(
            receiving_core_id, tensor_operand
        )
        if self.memory_manager.contains(tensor, receiving_top_instance):
            return -1, 0, 0, 0, 0, False, -1, None
        ################################# STEP 1 #################################
        # Get the top instance storing the tensor
        # If a sending core id is provided, we get the instance of that core.
        # Else, we find the instance where the tensor has been stored the longest
        if sending_core_id is not None:
            storing_instance = self.get_top_instance_of_core(
                sending_core_id, tensor.memory_operand
            )
            assert self.contains_tensor(tensor, storing_instance)
            available_since_timestep = (
                self.memory_manager.top_instance_available_since_timestep[
                    storing_instance
                ][tensor.equality_hash()]
            )
        else:
            (
                instances_storing_tensor,
                available_since_timesteps,
            ) = self.find_tensor_in_top_instances(tensor)
            # Pick the core that has stored the tensor the longest
            available_since_timestep = min(available_since_timesteps.values())
            storing_instance = next(
                (
                    top_instance
                    for (top_instance, timestep) in available_since_timesteps.items()
                    if timestep == available_since_timestep
                )
            )
        ################################# STEP 2 #################################
        # The receiver core has enough space to store the tensor.
        enough_space_timestep = self.memory_manager.get_timestep_for_tensor_addition(
            tensor,    
            receiving_core_id,
            available_since_timestep,
            memory_op=tensor_operand,
        )

        ################################# STEP 3 #################################
        # Make space on the receiving core by evicting tensors if there was never enough space.
        (
            evictions_complete_timestep,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
        ) = self.make_space_for(
            dbg_memTile_file=dbg_memTile_file,
            tensor=tensor,  
            core=receiving_core,
            memory_op=tensor_operand,
            timestep=enough_space_timestep,
            links_printing_file=links_printing_file,
            tensors_to_avoid_evicting=non_evictable_tensors,
        )
        ################################# STEP 4 #################################
        # The links between sender and receiver have a long enough idle window.
        sender_cores = self.memory_manager.cores_per_top_instance[storing_instance]
        # TODO If the storing_instance is a shared instance across more than one core,
        # TODO there will be multiple possible cores to transfer between.
        # TODO For now, we take the first one
        sender_core = sender_cores[0]

        # Aya: If the data is coming from the offchip, I will run my heuristic to decide if I should go through the memTile or directly through the offchip
        came_from_offchip = sender_core.id == self.offchip_core_id

        ######### Aya: added the following heuristic to decide when the tool should explore the usage of memTile and when it should not
        use_memTile_flag_status = 0
        memTile_core = []
        if came_from_offchip:
            # retreieve the memTile in the same column as that of the receiver core and send it to the heuristic function
            cores_without_offchip = []
            for one_core in self.cores:
                if one_core.id is not self.offchip_core_id:
                    cores_without_offchip.append(one_core)
            accelerator_cores_array = np.asarray(cores_without_offchip).reshape((self.nb_rows, self.nb_cols), order="C")
            for col in accelerator_cores_array.T:
                if receiving_core in col:
                    for one_core in col:
                        if(one_core.core_type == 1):
                            memTile_core = one_core
            
            # Aya: use_memTile_flag_status can be one of three values: 
                # 0: indicating that we have enough offchip channels 
                # 1: indicating that the data is already present in the memTile and we should directly tansfer it from there
                # 2: indicating that the data is not present in the memTile and there is only one available offchip channel, so do offchip -> memTile then memTile -> core 
            use_memTile_flag_status = self.memTile_heuristic(memTile_core, memTile_flag, tensor, tensor_operand, sender_core, receiving_core, evictions_complete_timestep)

        if use_memTile_flag_status == 1: 
            actual_sender_core = memTile_core
            actual_receiving_core = receiving_core
        elif use_memTile_flag_status == 2:   
            # then we will return this information so that schedule_graph calls transfer_tensor_to_core again between the memTile and the original receving_core 
            actual_sender_core = sender_core
            actual_receiving_core = memTile_core
        else:  # for anything else, just go with the original sender
            actual_sender_core = sender_core
            actual_receiving_core = receiving_core      
        ############################################  end of Aya's memTile heuristic

        links = self.communication_manager.get_links_for_pair(  
            actual_sender_core, actual_receiving_core
        )
        links_nested = []
        for path in links:
            if hasattr(path, '__iter__'):
                links_nested.append({link: link.bandwidth for link in path})  # Aya: to support multiple parallel paths between a pair of cores where each path could be made of multiple links (i.e., if the pair of cores are not directly connected)
            else:
                links_nested.append({path: path.bandwidth})
        # links = {link: link.bandwidth for link in links} # added for broadcasting
        links = links_nested

        if use_memTile_flag_status == 2 and len(future_tensors) > 0:
            # Aya: since we will transfer through the memTile and we will prefetch multiple tensors, 
                # we need to calculate the time needed to add those future tensors
            for tens, tens_operand in zip(future_tensors, future_tensors_operands): 
                temp_time_step = evictions_complete_timestep
                enough_space_timestep = self.memory_manager.get_timestep_for_tensor_addition(
                    tens,
                    actual_receiving_core.id,
                    temp_time_step,
                    memory_op=tens_operand,  
                )
               
                #tensors_to_avoid_eviction = list(set().union(future_tensors, list(non_evictable_tensors)))
                tensors_to_avoid_eviction = all_future_tensors[all_future_tensors.index(future_tensors[0]): len(future_tensors) + 16]
                (
                    evictions_complete_timestep,
                    eviction_link_energy_cost,
                    eviction_memory_energy_cost,
                ) = self.make_space_for(
                    dbg_memTile_file=dbg_memTile_file,
                    tensor=tens,
                    core=actual_receiving_core,
                    memory_op=tens_operand,  
                    timestep=enough_space_timestep,
                    links_printing_file=links_printing_file,
                    tensors_to_avoid_evicting=tensors_to_avoid_eviction,  # Aya: TODO: What is the best thing to pass here...
                )
            ##########################

        # Aya: Pass the list of future_tensors instead of the single tensor to prefetch multiple tensors to the memTile
            transfer_start, transfer_duration, chosen_links, all_links_transfer_start_end = self.communication_manager.get_links_idle_window(
                links, evictions_complete_timestep, future_tensors, actual_sender_core, actual_receiving_core
            ) # [tensor,]
        else:
            transfer_start, transfer_duration, chosen_links, all_links_transfer_start_end = self.communication_manager.get_links_idle_window(
                links, evictions_complete_timestep, [tensor,], actual_sender_core, actual_receiving_core
            )
    
        transfer_end = transfer_start + transfer_duration
        ################################# STEP 5 #################################
        # Spawn the tensor on the receiving core
        # Aya
        if use_memTile_flag_status == 2 and len(future_tensors) > 0:
            # loop over future_tensors and call spawn for each of them
            for t in future_tensors:
                # first calculate the transfer duration for one t
                if hasattr(chosen_links, '__iter__'):
                    transfer_duration = max([ceil(t.size / link.bandwidth) for link in chosen_links])
                else:
                    transfer_duration = t.size / chosen_links.bandwidth

                transfer_end = transfer_start + transfer_duration
                self.spawn(t, actual_receiving_core, tensor_operand, transfer_start, transfer_end)
                transfer_start = transfer_end
        else:
            self.spawn(tensor, actual_receiving_core, tensor_operand, transfer_start, transfer_end)
        ################################# STEP 6 #################################
        # Update the links involved in the communication and get the transfer energy cost
        # Aya
        if use_memTile_flag_status == 2 and len(future_tensors) > 0:
            (
                transfer_link_energy_cost,
                transfer_memory_energy_cost,
            ) = self.communication_manager.update_links(
                future_tensors,
                actual_sender_core.id,
                actual_receiving_core.id,
                tensor_operand,
                transfer_start,
                transfer_duration,
                chosen_links,
            )
        else:
            (
                transfer_link_energy_cost,
                transfer_memory_energy_cost,
            ) = self.communication_manager.update_links(
                [tensor],  # Aya: changed the function to accept a list of tensors
                actual_sender_core.id,
                actual_receiving_core.id,
                tensor_operand,
                transfer_start,
                transfer_duration,
                chosen_links,
            )
        ################################# STEP 7 #################################
        # Remove the transfered tensor from the sender core (excluding DRAM , Aya: and excluding memTile)
        # if it is no longer needed.
        if actual_sender_core.id == self.offchip_core_id or actual_sender_core.core_type == 1:  # Aya: added the memTile to also not remove from it
            pass
        # Don't remove it from the producing core 
        else:
            not_on_producing_core = actual_sender_core.id != tensor.origin.core_allocation  
            if (storing_instance not in tensor.instance_priorities) or (
                not_on_producing_core and tensor.instance_priorities[storing_instance] == 0
            ):
                self.remove(
                    tensor,
                    actual_sender_core,
                    tensor.memory_operand,
                    transfer_end,
                    links_printing_file,
                    dbg_memTile_file,
                    write_back_to_offchip=False,
                )

         ################################# STEP 8 #################################
        # Give back flag that signals if the tensor came from offchip
        # Aya: moved it to happen above to run the heuristic of memTile to decide if I should go through the memTile or directly from the offchip
       
        with open(dbg_memTile_file, "a") as ff:
            print("\t----------------------------", file=ff)
            print("The value of came_from_offchip is {} and the value of use_memTile_flag_status is {} and the end of the transfer is {}".format(came_from_offchip, use_memTile_flag_status, transfer_end), file=ff)
            print("\t----------------------------", file=ff)

        return (
            transfer_end,
            transfer_link_energy_cost,
            transfer_memory_energy_cost,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
            came_from_offchip,
            use_memTile_flag_status,
            memTile_core, # Aya: schedule_graph function will use this core only if use_memTile_flag_status is 2
        )

    def get_memory_energy_cost_of_transfer(
        self,
        tensor: Tensor,
        sender: Core or int,
        receiver: Core or int,
        sender_memory_operand: str,
        receiver_memory_operand: str,
    ):
        # Convert given sender and receiver to Core object if given as ids
        if isinstance(sender, int):
            sender = self.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.get_core(receiver)

        # Get the top level of output memory for the sender and the top level of input memory for the consumer_operand
        sender_top_memory_level = sender.memory_hierarchy.get_operand_top_level(
            sender_memory_operand
        )
        receiver_top_memory_level = receiver.memory_hierarchy.get_operand_top_level(
            receiver_memory_operand
        )
        # Sender memory energy
        nb_sender_memory_reads_for_data = ceil(
            tensor.size / sender_top_memory_level.read_bw
        )
        sender_energy = (
            sender_top_memory_level.read_energy * nb_sender_memory_reads_for_data
        )
        # Receiver memory energy
        nb_receiver_memory_writes_for_data = ceil(
            tensor.size / receiver_top_memory_level.write_bw
        )
        receiver_energy = (
            receiver_top_memory_level.write_energy * nb_receiver_memory_writes_for_data
        )

        return sender_energy + receiver_energy

    def block_offchip_links(
        self, too_large_operands, core_id, start_timestep, duration, cn
    ) -> int:
        return self.communication_manager.block_offchip_links(
            too_large_operands, core_id, start_timestep, duration, cn
        )

    def contains_tensor(self, tensor: Tensor, top_instance):
        if isinstance(top_instance, int):  # assume core id
            memory_op = tensor.memory_operand
            top_instance = self.get_top_instance_of_core(top_instance, memory_op)

        return self.memory_manager.contains(tensor, top_instance)

    def find_tensor(self, tensor: Tensor):
        return self.memory_manager.find_tensor(tensor)

    def find_tensor_in_top_instances(self, tensor: Tensor):
        return self.memory_manager.find_tensor_in_top_instances(tensor)

    def has_shared_memory(self, core_id_a, core_id_b, mem_op_a, mem_op_b):
        """Check whether two cores have a shared top level memory instance for a given memory operand.

        Args:
            core_id_a (int): The first core id.
            core_id_b (int): The second core id.
            mem_op_a (str): The memory operand for the tensor in core a.
            mem_op_b (str): The memory operand for the tensor in core b.
        """
        core_a = self.get_core(core_id_a)
        core_b = self.get_core(core_id_b)
        top_memory_instance_a = next(
            (
                ml.memory_instance
                for ml, out_degree in core_a.memory_hierarchy.out_degree()
                if out_degree == 0 and mem_op_a in ml.operands
            )
        )
        top_memory_instance_b = next(
            (
                ml.memory_instance
                for ml, out_degree in core_b.memory_hierarchy.out_degree()
                if out_degree == 0 and mem_op_b in ml.operands
            )
        )
        return top_memory_instance_a is top_memory_instance_b

    def get_top_instances_of_core(self, core_id):
        core = self.get_core(core_id)
        top_instances = self.memory_manager.top_instances[core]
        return top_instances

    def get_top_instance_of_core(self, core, mem_op):
        if isinstance(core, int):
            core = self.get_core(core)
        top_instances = self.memory_manager.top_instances[core]
        for instance in top_instances:
            core_idx = self.memory_manager.cores_per_top_instance[instance].index(core)
            instance_mem_ops = self.memory_manager.memory_operands_per_top_instance[
                instance
            ][core_idx]
            if mem_op in instance_mem_ops:
                return instance
        raise ValueError(f"No top instance for {core} with memory operand {mem_op}.")
