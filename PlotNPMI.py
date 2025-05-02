import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mpltPath

### Data for sulfones from wet lab ###
sf_name = 'all-figures_nprs-removedNP.csv'
#pull in data as DF
sf_df = pd.read_csv(f'{sf_name}')
#pull the NPRs from prepared column data; store to lists
sf_x1s = sf_df['X1s'].to_list()
sf_y1s = sf_df['Y1s'].to_list()

### Generic biphenyl from CHEMBL DB ###
gen_name = 'chembel_nprs_last.csv'
gen_df = pd.read_csv(f'{gen_name}')

#pull the NPRs from prepared column data; store to lists
gen_x1s = gen_df['X1s'].to_list()
gen_y1s = gen_df['Y1s'].to_list()

#Initialize plotting
plt.figure(figsize=(18.2,16))
plt.rcParams['axes.linewidth'] = 3.0
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

#Format the Disk/Linear/Sphere triangle
#vertices of the triangle plot
x1, y1 = [0.5, 0], [0.5, 1]
x2, y2 = [0.5, 1], [0.5, 1]
x3, y3 = [0, 1], [1, 1]

#draw gray lines connecting the above
plt.plot(x1,y1,x2,y2,x3,y3,
    c='black', lw=3)

#add some text blocks to each corner
plt.text(0, 1.02, s='Linear', fontsize=55, ha='left',va='center', fontweight='bold')
plt.text(1, 1.02, s='Sphere', fontsize=55, ha='right',va='center', fontweight='bold')
plt.text(0.5, 0.472, s='Disk', fontsize=55, ha='center',va='bottom', fontweight='bold')

#plot the gen data
gen_plot = plt.scatter(gen_x1s, gen_y1s,
            color=(181/255, 182/255, 139/255), label='Chembl',
            linewidths=0.1, alpha=0.5, s=110, edgecolors=(181/255, 
            182/255, 139/255), linewidth=2.0)

#plot the sf data
sf_plot = plt.scatter(sf_x1s, sf_y1s,
            color=(1, 0.24, 0.06), label='wet lab',
            linewidths=0.1, alpha=0.7, s=110, edgecolors=(1, 
            0.24, 0.06), linewidth=2.0)

#axis formatting
plt.xlabel('PMI(x)', fontsize=70, fontweight='bold', labelpad=15)
plt.ylabel('PMI(y)', fontsize=70, fontweight='bold', labelpad=15)

plt.tick_params(axis='x', labelsize=55, width=8.0, length=10.0)
plt.tick_params(axis='y', labelsize=55, width=8.0, length=10.0)

plt.savefig('chembl-wet_lab_data_24_04_2025.png', bbox_inches="tight")

## Section for defining the polygon area as shwon in figure S10
# Define the vertices of your polygon section
# Replace these coordinates with the actual vertices of your marked section
polygon_vertices = [(0.1, 1.0), (0.4, 0.95), (0.43, 0.58)]
# Create a Path object
polygon_path = mpltPath.Path(polygon_vertices)
polygon_x, polygon_y = zip(*polygon_vertices)
polygon_x = list(polygon_x) + [polygon_x[0]]
polygon_y = list(polygon_y) + [polygon_y[0]]
plt.plot(polygon_x, polygon_y, color='blue', linewidth=4, linestyle='--', label='Polygon Path')
# Assuming your data points are in a list or array like:
data_points = gen_df[['X1s', 'Y1s']].values.tolist()
# Convert to numpy array if not already
data_points = np.array(data_points)
# Check which points are inside the polygon
inside = polygon_path.contains_points(data_points)
# Count the number of points inside
num_points_inside = np.sum(inside)
print(f"Number of points inside the section: {num_points_inside}")
# If you need to extract the actual points inside:
points_inside = data_points[inside]
# plt.show()



