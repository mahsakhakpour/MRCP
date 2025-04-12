import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.patches import Circle
import time

def get_user_input():
    print("\n*** Maximum Range Count Problem *** \n ** Mahsa Khakpour **")

    points = []
    print("\nEnter your 2D points (x,y). Type 'done' when finished:")
    while True:
        user_input = input("Enter point (format: x,y): ").strip()
        if user_input.lower() == 'done':
            if len(points) < 2:
                print("Please enter at least 2 points!")
                continue
            break
        try:
            x, y = map(float, user_input.split(','))
            points.append([x, y])
            print(f"Added point ({x:.2f}, {y:.2f}) | Total points: {len(points)}")
        except ValueError:
            print("Invalid format! Please enter as 'x,y' (e.g., '1.0,2.0')")

    while True:
        try:
            eps = float(input("\nEnter maximum cluster distance (eps): "))
            min_samples = int(input("Enter minimum points per cluster: "))
            radius = float(input("Enter query circle radius: "))
            if eps > 0 and min_samples > 0 and radius > 0:
                break
            print("All values must be positive numbers!")
        except ValueError:
            print("Please enter valid numbers!")

    return np.array(points), eps, min_samples, radius

def sliding_circle_algorithm(points, radius, cluster_points):
    if len(cluster_points) == 0:
        return None, 0

    best_center = None
    max_count = 0
    
    for start_point in cluster_points:
        current_center = start_point.copy()
        scanned = np.zeros(len(cluster_points), dtype=bool)
        local_max = 0
        local_best = None
        
        for _ in range(100):
            distances = np.linalg.norm(cluster_points - current_center, axis=1)
            in_circle = distances <= radius
            new_points = in_circle & ~scanned
            current_count = np.sum(new_points)
            
            if current_count > local_max:
                local_max = current_count
                local_best = current_center.copy()
            
            scanned[in_circle] = True
            unscanned_indices = np.where(~scanned)[0]
            if len(unscanned_indices) == 0:
                break
            
            unscanned_points = cluster_points[unscanned_indices]
            dist_to_unscanned = np.linalg.norm(unscanned_points - current_center, axis=1)
            nearest_idx = np.argmin(dist_to_unscanned)
            nearest_point = unscanned_points[nearest_idx]
            
            direction = nearest_point - current_center
            step_size = min(radius/4, np.linalg.norm(direction))
            if step_size > 0:
                current_center += direction * (step_size / np.linalg.norm(direction))
            else:
                break
        
        if local_max > max_count:
            max_count = local_max
            best_center = local_best

    return best_center, max_count

def brute_force_algorithm(points, radius, cluster_points, resolution=0.05):
    if len(cluster_points) == 0:
        return None, 0
    
    padding = radius * 1.5
    x_min, y_min = np.min(cluster_points, axis=0) - padding
    x_max, y_max = np.max(cluster_points, axis=0) + padding
    step = radius * resolution

    best_center = None
    max_count = 0
    
    x_grid = np.arange(x_min, x_max + step, step)
    y_grid = np.arange(y_min, y_max + step, step)
    
    for x in x_grid:
        for y in y_grid:
            center = np.array([x, y])
            distances = np.linalg.norm(cluster_points - center, axis=1)
            count = np.sum(distances <= radius)
            if count > max_count:
                max_count = count
                best_center = center

    return best_center, max_count

def visualize_results(points, labels, sliding_results, brute_results, radius, title):
    plt.figure(figsize=(14, 10))
    clusters = set(labels) - {-1}
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters))) if clusters else ['blue']

    # First count points in each cluster
    cluster_counts = {}
    for cluster_id in clusters:
        cluster_counts[cluster_id] = np.sum(labels == cluster_id)
    
    for i, cluster_id in enumerate(clusters):
        cluster_points = points[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i], 
                    label=f'Cluster ({cluster_counts[cluster_id]} pts)', 
                    alpha=0.7, s=80)

    if -1 in labels:
        noise_points = points[labels == -1]
        noise_count = len(noise_points)
        plt.scatter(noise_points[:, 0], noise_points[:, 1],
                    c='gray', marker='x', 
                    label=f'Noise ({noise_count} pts)', 
                    alpha=0.5)

    # Draw sliding circles (red)
    best_sliding = max(sliding_results, key=lambda x: x[2], default=None)
    for cid, center, count in sliding_results:
        if (cid, center, count) != best_sliding:
            plt.gca().add_patch(Circle(center, radius, fill=False, color='red',
                                     linewidth=2, linestyle='-'))
            plt.text(center[0], center[1], f'S{count}', color='red', ha='center')

    # Draw brute force circles (green)
    best_brute = max(brute_results, key=lambda x: x[2], default=None)
    for cid, center, count in brute_results:
        if (cid, center, count) != best_brute:
            plt.gca().add_patch(Circle(center, radius, fill=False, color='green',
                                     linewidth=2, linestyle='--'))
            plt.text(center[0], center[1], f'B{count}', color='green', ha='center')

    # Draw optimal brute (dashed black)
    if best_brute:
        _, center, count = best_brute
        plt.gca().add_patch(Circle(center, radius, fill=False, color='black',
                                 linewidth=3, linestyle='--', label=f'Brute Optimal ({count} pts)'))
        plt.text(center[0], center[1], f'★{count}', color='black', fontsize=12, ha='center')

    # Draw optimal sliding (solid black)
    if best_sliding:
        _, center, count = best_sliding
        plt.gca().add_patch(Circle(center, radius, fill=False, color='black',
                                 linewidth=3, linestyle='-', label=f'Sliding Optimal ({count} pts)'))
        plt.text(center[0], center[1], f'★{count}', color='black', fontsize=12, ha='center', va='bottom')

    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.grid(True)
    
    # Get existing handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Filter out any duplicate labels
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    plt.legend(handles=unique_handles, labels=unique_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def run_algorithm(name, func, points, radius, cluster_points):
    results = []
    total_time = 0
    for cid, cpoints in cluster_points.items():
        if len(cpoints) < 2:
            continue
        start = time.time()
        center, count = func(points, radius, cpoints)
        duration = time.time() - start
        total_time += duration
        results.append((cid, center, count))
        print(f"  {name} - Cluster {cid}: {count} points at ({center[0]:.1f}, {center[1]:.1f}) in {duration:.4f} sec")
    return results, total_time

def main():
    points, eps, min_samples, radius = get_user_input()
    print("\nRunning DBSCAN clustering...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clusters = set(labels) - {-1}
    cluster_points = {cid: points[labels == cid] for cid in clusters}
    if not clusters:
        cluster_points = {-1: points}

    print("\n🔴 Running Sliding Circle Algorithm...")
    sliding_results, sliding_time = run_algorithm("Sliding", sliding_circle_algorithm, points, radius, cluster_points)

    print("\n🟢 Running Brute Force Algorithm...")
    brute_results, brute_time = run_algorithm("Brute", brute_force_algorithm, points, radius, cluster_points)

    print("\n⏱ Timing Summary:")
    print(f"  🔴 Sliding Circle: {sliding_time:.4f} seconds")
    print(f"  🟢 Brute Force   : {brute_time:.4f} seconds")

    visualize_results(points, labels, sliding_results, brute_results, radius, "Sliding vs. Brute Force – Max Circle Coverage")

if __name__ == "__main__":
    main()
