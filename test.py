import numpy as np

# Tạo dữ liệu giả
dataset = np.array([
    [1, 2],  # t=0
    [2, 3],  # t=1 
    [3, 4],  # t=2
    [4, 5],  # t=3
    [5, 6],  # t=4
    [6, 7]   # t=5
])

# Giả lập params
params = {'history': 2}

# Test function
def test_window_transform():
    data = []
    labels = []
    history = params['history']
    
    for i in range(history, len(dataset) - 1):
        window = dataset[i-history:i]
        target = dataset[i+1]
        print(f"\nWindow {i}:")
        print("Input window:", window)
        print("Target:", target)
        data.append(window)
        labels.append(target)
    
    return np.array(data), np.array(labels)

X, y = test_window_transform()
print("\nFinal X shape:", X.shape)
print("Final y shape:", y.shape)