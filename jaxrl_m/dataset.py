import numpy as np
from jaxrl_m.typing import Data
from flax.core.frozen_dict import FrozenDict
from jax import tree_util
import jax
import jax.numpy as jnp

def get_size(data: Data) -> int:
    sizes = jax.tree.map(lambda arr: len(arr), data)
    return max(tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """
    A class for storing (and retrieving batches of) data in nested dictionary format.

    Example:
        dataset = Dataset({
            'observations': {
                'image': np.random.randn(100, 28, 28, 1),
                'state': np.random.randn(100, 4),
            },
            'actions': np.random.randn(100, 2),
        })

        batch = dataset.sample(32)
        # Batch will have nested shape: {
        # 'observations': {'image': (32, 28, 28, 1), 'state': (32, 4)},
        # 'actions': (32, 2)
        # }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

    def sample(self, batch_size: int, indx=None):
        """
        Sample a batch of data from the dataset. Use `indx` to specify a specific
        set of indices to retrieve. Otherwise, a random sample will be drawn.

        Returns a dictionary with the same structure as the original dataset.
        """
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)

    def get_subset(self, indx):
        return jax.tree.map(lambda arr: arr[indx], self._dict)


class ReplayBuffer(Dataset):
    """
    Dataset where data is added to the buffer.

    Example:
        example_transition = {
            'observations': {
                'image': np.random.randn(28, 28, 1),
                'state': np.random.randn(4),
            },
            'actions': np.random.randn(2),
        }
        buffer = ReplayBuffer.create(example_transition, size=1000)
        buffer.add_transition(example_transition)
        batch = buffer.sample(32)

    """

    @classmethod
    def create(cls, transition: Data, size: int):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree.map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: dict, size: int):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree.map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree.map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)
    
    def get_all(self):
        
        batch = jax.tree.map(lambda x: x[:self.size], self._dict)
        return jax.tree.map(lambda x: jax.device_put(x), batch)
    
    def reset(self):
        self.size = 0
        self.pointer = 0
        return self
    
    
class ActorReplayBuffer(ReplayBuffer):
    
    def get_all(self):
        
        
            batch = jax.tree.map(lambda x: x[:self.size], self._dict)
            batch = jax.tree.map(lambda x: jnp.pad(x, ((0, self.max_size - self.size),) + ((0, 0),) * (x.ndim - 1), mode='constant'), batch)
        
            return jax.tree.map(lambda x: jax.device_put(x), batch)
        
        