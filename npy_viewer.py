import matplotlib.pyplot as plt
import numpy as np

data = np.load('/Users/jackhilton-jones/Proper-Research/benchmark_no_contact_fast_serial/poses_used.npy')

# If the data is an image (2D or 3D array)
print("Data Shape:", data.shape)
print("Data Type:", data.dtype)

# View the actual contents
print(data)
