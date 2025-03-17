import numpy as np


class Policy:
    """
    This class is the interface where the evaluation scripts communicate with your trained agent.

    You can initialize your model and load weights in the __init__ function. At each environment interactions,
    the batched observation `obs`, a numpy array with shape (Batch Size, Obs Dim), will be passed into the __call__
    function. You need to generate the action, a numpy array with shape (Batch Size, Act Dim=2), and return it.

    Do not change the name of this class.

    Please do not import any external package.
    """

    # FILLED YOUR PREFERRED NAME & UID HERE!
    CREATOR_NAME = "Zhenghao Peng"  # Your preferred name here in a string
    CREATOR_UID = "0000000000"  # Your UID here in a string

    def __init__(self):
        # Load weights here
        pass

    def reset(self, done_batch=None):
        """
        Optionally reset the latent state of your agent, if any.

        Args:
            done_batch: an array with shape (batch_size,) in vectorized environment or a boolean in single environment.
            True represents the latent state of this episode should be reset.
            If it's None, you should reset the latent state for all episodes.

        Returns:
            None
        """
        pass

    def __call__(self, obs):
        return np.zeros((len(obs), 2))
