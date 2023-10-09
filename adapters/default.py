from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from helios_rl.encoders.prior_actions_encoded import PriorActionsEncoder

class DefaultAdapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self):
        # TODO: Update this based on the current problem, each requires preset knowledge of all possible states/actions/objects
        # - Possible States
        # - Possible Actions
        # - Prior Actions
        # - Possible Objects
    
        # Minigrid states already encoded, but could try following as well:

        # - Initialise encoder based on all prior actions
        all_possible_actions = [i for i in range(7)]
        self.encoder = PriorActionsEncoder(all_possible_actions)
    
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
        # State must be tensor to not need encoding
        # Encode to Tensor for agents
        if encode:
            state_encoded = state #self.encoder.encode(episode_action_history=episode_action_history)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in DefaultAdapter._cached_state_idx):
                    DefaultAdapter._cached_state_idx[sent] = len(DefaultAdapter._cached_state_idx)
                state_indexed.append(DefaultAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded