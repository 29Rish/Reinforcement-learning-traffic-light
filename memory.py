import random

class Memory:
    def __init__(self, size_max, size_min):
        # Initialize a memory buffer to store samples.
        # size_max: The maximum allowed size of the memory buffer.
        # size_min: The minimum size required to perform sampling.
        self._samples = []  # List to hold the stored samples.
        self._size_max = size_max  # Maximum size of the memory buffer.
        self._size_min = size_min  # Minimum size required for sampling.

    def add_sample(self, sample):
        # Add a new sample to the memory buffer.
        self._samples.append(sample)  # Append the new sample to the list.
        
        # If the memory exceeds its maximum capacity, remove the oldest sample (FIFO behavior).
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # Remove the first (oldest) element to maintain the buffer size.

    def get_samples(self, n):
        # Retrieve a random subset of samples from the memory buffer.
        # n: The number of samples to retrieve.

        # If the memory buffer size is below the minimum threshold, return an empty list.
        if self._size_now() < self._size_min:
            return []

        # If the requested number of samples (n) exceeds the current memory size, return all samples.
        if n > self._size_now():
            return random.sample(self._samples, self._size_now())
        else:
            # Otherwise, return a random subset of `n` samples.
            return random.sample(self._samples, n)

    def _size_now(self):
        # Private helper method to get the current size of the memory buffer.
        return len(self._samples)
