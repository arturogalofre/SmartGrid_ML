import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.getlogin()
os.getcwd()


#########################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.linspace(-6, 6, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sigmoid Plot')
plt.grid()
plt.show()

#########################


# Define the Gaussian function

def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))


# Generate x values
x = np.linspace(-10, 10, 100)

# Calculate y values for the two different standard deviation values
y1 = gaussian(x, 0, 1)
y2 = gaussian(x, 0, 2)

# Plot the two curves
plt.plot(x, y1, marker='.', linestyle='-', label='Std. Dev = 1')
plt.plot(x, y2, marker='.', linestyle='-', label='Std. Dev = 2')

# Add grid, title and labels
plt.grid()
plt.title('Gaussian Probability Density Function')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

# Show the plot
plt.show()

###############################


# Load the data from csv
df = pd.read_csv("PMU01_140701.csv")

# Select the rows where 'VCLPM:Magnitude' is less than 199000
df_filtered = df[df['VCLPM:Magnitude'] < 199000]

# Select the columns 'Timestamp', 'VALPM:Magnitude', 'VBLPM:Magnitude', 'VCLPM:Magnitude'
df_selected = df_filtered[[
    'Timestamp', 'VALPM:Magnitude', 'VBLPM:Magnitude', 'VCLPM:Magnitude']]

# Print the data
print(df_selected)


##################################
