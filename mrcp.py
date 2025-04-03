import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.patches import Circle
import time

def get_user_input():
    """Get all required inputs from the user with validation"""
    print("\n*** Maximum Range Count Problem *** \n ** Mahsa Khakpour **")
    
    # Get dataset
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
    
    # Get parameters with validation
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

def sliding_circle_algorithm(points, radius):
    """Improved sliding circle algorithm with multi-start strategy"""
    if len(points) == 0:
        return None, 0
    
    best_center = None
    max_count = 0
    
    # Try starting from each point to ensure coverage
    for start_point in points:
        current_center = start_point.copy()
        scanned = np.zeros(len(points), dtype=bool)
        local_max = 0
        local_best = None
        
        for _ in range(100):  # Max iterations per start point
            # Find points in current circle
            distances = np.linalg.norm(points - current_center, axis=1)
            in_circle = distances <= radius
            
            # Count unscanned points
            new_points = in_circle & ~scanned
            current_count = np.sum(new_points)
            
            # Update local best
            if current_count > local_max:
                local_max = current_count
                local_best = current_center.copy()
            
            # Mark points as scanned
            scanned[in_circle] = True
            
            # Find nearest unscanned point
            unscanned_indices = np.where(~scanned)[0]
            if len(unscanned_indices) == 0:
                break
                
            unscanned_points = points[unscanned_indices]
            dist_to_unscanned = np.linalg.norm(unscanned_points - current_center, axis=1)
            nearest_idx = np.argmin(dist_to_unscanned)
            nearest_point = unscanned_points[nearest_idx]
            
            # Move with adaptive step size
            direction = nearest_point - current_center
            step_size = min(radius/4, np.linalg.norm(direction))
            if step_size > 0:
                current_center += direction * (step_size / np.linalg.norm(direction))
            else:
                break
        
        # Update global best
        if local_max > max_count:
            max_count = local_max
            best_center = local_best
    
    return best_center, max_count

def brute_force_algorithm(points, radius, resolution=0.05):
    """High-accuracy brute-force validation"""
    if len(points) == 0:
        return None, 0
    
    # Create expanded search space
    padding = radius * 1.5
    x_min, y_min = np.min(points, axis=0) - padding
    x_max, y_max = np.max(points, axis=0) + padding
    step = radius * resolution
    
    best_center = None
    max_count = 0
    
    # Vectorized grid evaluation
    x_grid = np.arange(x_min, x_max + step, step)
    y_grid = np.arange(y_min, y_max + step, step)
    
    for x in x_grid:
        for y in y_grid:
            center = np.array([x, y])
            distances = np.linalg.norm(points - center, axis=1)
            count = np.sum(distances <= radius)
            
            if count > max_count:
                max_count = count
                best_center = center
    
    return best_center, max_count

def visualize_results(points, labels, sliding_results, brute_results, radius):
    """Enhanced visualization with clear optimal circles"""
    plt.figure(figsize=(14, 10))
    
    # Plot clusters
    clusters = set(labels) - {-1}
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters))) if clusters else ['blue']
    for i, cluster_id in enumerate(clusters):
        cluster_points = points[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   color=colors[i], label=f'Cluster {cluster_id}', alpha=0.7, s=50)
    
    # Plot noise points if any
    if -1 in labels:
        noise_points = points[labels == -1]
        plt.scatter(noise_points[:, 0], noise_points[:, 1],
                   c='gray', marker='x', label='Noise', alpha=0.5)

    # ===== SLIDING CIRCLE RESULTS =====
    if sliding_results:
        best_slide = max(sliding_results, key=lambda x: x[2])
        
        # Optimal sliding circle (thick red)
        plt.gca().add_patch(Circle(best_slide[1], radius,
                           fill=False, color='red', linewidth=3,
                           label=f'Sliding Optimal: {best_slide[2]} pts'))
        plt.text(best_slide[1][0], best_slide[1][1], 
                f'SLIDE: {best_slide[2]}', ha='center', va='center',
                color='red', fontsize=10, weight='bold')
        
        # Other sliding circles (thin blue)
        for res in sliding_results:
            if res[2] != best_slide[2]:
                plt.gca().add_patch(Circle(res[1], radius,
                                         fill=False, color='blue', linewidth=1,
                                         linestyle=':', alpha=0.4))
                plt.text(res[1][0], res[1][1], str(res[2]),
                        ha='center', va='center', color='blue', fontsize=8)

    # ===== BRUTE-FORCE RESULTS =====
    if brute_results:
        best_brute = max(brute_results, key=lambda x: x[2])
        
        # Optimal brute circle (thick green)
        plt.gca().add_patch(Circle(best_brute[1], radius,
                           fill=False, color='green', linewidth=2,
                           linestyle='--', alpha=0.8,
                           label=f'Brute Optimal: {best_brute[2]} pts'))
        plt.text(best_brute[1][0], best_brute[1][1], 
                f'BRUTE: {best_brute[2]}', ha='center', va='center',
                color='green', fontsize=10, weight='bold')
        
        # Other brute circles (thin green)
        for res in brute_results:
            if res[2] != best_brute[2]:
                plt.gca().add_patch(Circle(res[1], radius,
                                         fill=False, color='green', linewidth=0.5,
                                         linestyle='--', alpha=0.3))

    plt.title('Optimal Circle Placement Comparison\n(Red: Sliding Algorithm, Green: Brute-Force)',
             fontsize=12, pad=20)
    plt.xlabel('X coordinate', fontsize=10)
    plt.ylabel('Y coordinate', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def main():
    # Get user input
    points, eps, min_samples, radius = get_user_input()
    
    # DBSCAN Clustering
    print("\nRunning DBSCAN clustering...")
    start_time = time.time()
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    cluster_time = time.time() - start_time
    
    # Cluster analysis
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise} ({n_noise/len(points)*100:.1f}%)")
    print(f"  Clustering time: {cluster_time:.2f} seconds")

    # Process each cluster
    sliding_results = []
    brute_results = []
    clusters = set(labels) - {-1}
    
    if not clusters:
        print("\nNo clusters found - processing all points as one group")
        clusters = {-2}
        cluster_points = { -2: points }
    else:
        cluster_points = { cid: points[labels == cid] for cid in clusters }

    print("\nProcessing clusters...")
    for cid in clusters:
        cpoints = cluster_points[cid]
        print(f"\nCluster {cid if cid != -2 else 'All Points'}: {len(cpoints)} points")
        
        # Sliding Circle Algorithm
        start = time.time()
        s_center, s_count = sliding_circle_algorithm(cpoints, radius)
        s_time = time.time() - start
        if s_center is not None:
            sliding_results.append((cid, s_center, s_count))
            print(f"  Sliding Circle: {s_count} points at ({s_center[0]:.1f}, {s_center[1]:.1f})")
            print(f"  Time: {s_time:.2f} seconds")
        
        # Brute Force Algorithm
        start = time.time()
        b_center, b_count = brute_force_algorithm(cpoints, radius)
        b_time = time.time() - start
        if b_center is not None:
            brute_results.append((cid, b_center, b_count))
            print(f"  Brute Force: {b_count} points at ({b_center[0]:.1f}, {b_center[1]:.1f})")
            print(f"  Time: {b_time:.2f} seconds")
        
        # Compare results
        if s_center is not None and b_center is not None:
            diff = abs(s_count - b_count)
            if diff == 0:
                print("  ✅ Results match perfectly")
            elif diff <= 2:
                print(f"  ⚠️ Minor difference: {diff} points")
            else:
                print(f"  ❗ Significant difference: {diff} points")

    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(points, labels, sliding_results, brute_results, radius)

if __name__ == "__main__":
    main()