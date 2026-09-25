import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def verify_dataset(filepath='val_grid.npy'):
    data = np.load(filepath)
    n = data.shape[0]
    
    extent = [-0.8, 0.8, -0.8, 0.8]
    
    
    colors = ['#555555', '#000000', '#E63946', '#2A9D8F', '#457B9D']
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(data, origin='lower', extent=extent, cmap=cmap, vmin=-1, vmax=3)
    plt.title(r"Hénon–Heiles Exit Basins (${n}x{n}$)", fontsize=12, fontweight='bold')
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$y_0$")
    plt.colorbar(ticks=[-1, 0, 1, 2, 3], label="Exit Label")
    
    plt.subplot(1, 3, 2)
    labels, counts = np.unique(data, return_counts=True)
    
    label_names = { -1: 'Trapped (-1)', 0: 'Forbidden (0)', 
                    1: 'Exit 1 (Top)', 2: 'Exit 2 (Left)', 3: 'Exit 3 (Right)' }
    names = [label_names.get(l, str(l)) for l in labels]
    
    bars = plt.bar(names, counts, color=['#555555', '#111111', '#E63946', '#2A9D8F', '#457B9D'][:len(labels)])
    plt.title("Class Distribution Check", fontsize=12, fontweight='bold')
    plt.ylabel("Point Count")
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    total = data.size
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                 f"{height/total*100:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.subplot(1, 3, 3)
    mid_row = n // 2
    x_coords = np.linspace(-0.8, 0.8, n)
    slice_data = data[mid_row, :]
    
    plt.step(x_coords, slice_data, where='mid', color='purple', linewidth=1)
    plt.title(r"1D Slice at $y_0 = 0.0$", fontsize=12, fontweight='bold')
    plt.xlabel(r"$x_0$")
    plt.ylabel("Exit Class")
    plt.yticks([-1, 0, 1, 2, 3])
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    verify_dataset('val_grid.npy')