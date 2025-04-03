# Maximum Range Count Problem (MRCP) Solver

## Getting Started with MRCP Solver
This Python project implements algorithms to solve the "Maximum Range Count" problem, which involves finding the optimal placement of a circle with a given radius to cover the maximum number of points from a 2D dataset.

### Prerequisites
- Python 3.8+
- Required packages: NumPy, Matplotlib, scikit-learn

### Installation
1. Clone the repository:

git clone https://github.com/mahsakhakpour/mrcp.git
cd mrcp

Install dependencies:
pip install numpy matplotlib scikit-learn


Usage

Run the MRCP solver in interactive mode:

python mrcp.py

The program will:

Prompt for 2D points (format: x,y)

Request clustering parameters (eps, min_samples)

Ask for the query circle radius

Display analysis results with an interactive visualization


Example Execution

*** Maximum Range Count Problem ***
** Mahsa Khakpour **

Enter your 2D points (x,y). Type 'done' when finished:
Enter point (format: x,y): 1.0,2.0
Added point (1.00, 2.00) | Total points: 1
Enter point (format: x,y): done

Enter maximum cluster distance (eps): 5
Enter minimum points per cluster: 3
Enter query circle radius: 5

Output

Console output with detailed metrics

Visualization showing:

Cluster assignments (colored points)
Optimal circle placement
Performance comparisons
Interactive Matplotlib window

License
MIT License

Author
Mahsa Khakpour
mahsa54@gmail.com


