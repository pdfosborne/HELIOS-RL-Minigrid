import gymnasium as gym
import torch

class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self) -> None:
        """Initialize Engine"""
        self.Environment = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")

    def reset(self):
        """Fully reset the environment."""
        obs, _ = self.Environment.reset(seed=42)
        return torch.tensor(obs['image'])
    
    def step(self, state:any, action:any):
        """Enact an action."""           
        obs, reward, terminated, _, _ = self.Environment.step(action)
        # Only use rgb-array for observation (ignore direction)
        return torch.tensor(obs['image']), reward, terminated

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        # The agents have a discrete action space (A) of seven options representing 
        # [“turn left”, “turn right”, “move forward”, “pickup”, “drop”, “toggle”, “done”]. 
        legal_moves = [0,1,2,3,4,5,6] #self.Environment.legal_moves(obs)
        return legal_moves

