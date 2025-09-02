import numpy as np
import matplotlib.pyplot as plt

# Import the dataset from the file
S = np.genfromtxt("PCB.dt")

# Print the dataset
#print('PCB dataset :\n', S)

# To plot the results, take the dataset as input (consisting of pairs (x, y)) and the couple (w, b) defining the affine model.
def plot(data, reg_model, xlabel, ylabel, title = '', test  = "affine"):
	plt.figure()

	# Plot the ploints from the dataset.
	for [x, y] in data:
		plt.plot(x, y, 'bo')

	# Plot the linear model (or other models) from the regresssion.
	x = np.linspace(0, max(data[:, 0]), 100)
	if test == "affine":
		y = reg_model[0] * x + reg_model[1]
	elif test == "exp":
		y = np.exp(reg_model[0] * x + reg_model[1])
	elif test == "ln_final":
		y = reg_model[0] * np.sqrt(x) + reg_model[1]
	plt.plot(x, y, color='red')

	# Define the label of the axis, the title and save the file.
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(title)


# QUESTION 1
# A linear regression function, takes as input a dataset consisting of a list of pairs (x, y).
# Return a couple (w, b), parameter of the affine linear model resulting from the linear regression.
def lin_reg_1D(data):
	X = np.concatenate((data[:, 0].reshape(-1, 1), np.ones(len(data)).reshape(-1, 1)), axis=1)
	Y = data[:, 1].reshape(-1, 1)

	# Compute the linear model, as presented in the lecture note: (X^T X)^-1 X^T Y
	res = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

	return res

# Compute the Mean-Squared Error for a given dataset (1 Dimensional) and parameter of an affine function.
def MSE(data, reg_model):
	w, b = reg_model[0], reg_model[1]
	res = 0
	for [x, y] in data:
		res += (y - (w * x + b))**2
	return res / len(data)



# (Question 2) Compute the Mean-Squared Error for a given dataset (1 Dimensional) and parameter of for exp model.
def MSE_exp(data, reg_model):
	w, b = reg_model[0], reg_model[1]
	res = 0
	for [x, y] in data:
		res += (y - np.exp(w * x + b))**2
	return res / len(data)



# (Question 6) Compute the Mean-Squared Error for a given dataset (1 Dimensional) and parameter of for final model.
def MSE_sqrt(data, reg_model):
	w, b = reg_model[0], reg_model[1]
	res = 0
	for [x, y] in data:
		res += (y - np.exp(w *np.sqrt(x) + b))**2
	return res / len(data)



# Generate the plot for a basic linear regression over the PCB dataset.
linear_model = lin_reg_1D(S)
plot(S, linear_model, "years", "PCB", "LinearModel")
print('MSE of the linear model : ', MSE(S, linear_model))


# QUESTION 2 (And plto for QUESTION 4): 'exponential' model.
Sprime = np.array([[x, np.log(y)] for [x, y] in S])
exp_model = lin_reg_1D(Sprime)
print("QUESTION 2\nParameters of the exp model : ", exp_model)
plot(Sprime, exp_model, "years", "ln(PCB)", "Q4_ExpModel_Sprime")
plot(S, exp_model, "years", "PCB", "Q2_ExpModel_S", test = "exp")
print('MSE of the exp model : ', MSE_exp(S, exp_model))



# QUESTION 3: A plot to show a solution to Q3.
def Q3_plot():
	plt.figure()
	plt.title("Question 3")
	plt.xlabel("b")
	
	# Define the two terms.
	b = np.linspace(-2, 1, 1000)
	u, i = 0.4, np.exp(1.01)
	y1 = (u - np.exp(b))**2 + (i - np.exp(b))**2
	y2 = (np.log(u) - b)**2 + (np.log(i) - b)**2

	# Plot.
	plt.plot(b, y1, color='blue')
	plt.plot(b, y2, color='red')

	# Save file.
	plt.savefig("plot_Q3")
	
Q3_plot()



# QUESTION 5: Compute the coefficient of determination R^2
def compute_coef(data, exp_model, exp = True):
	ybar = np.mean(data[:, 1])
	w, b = exp_model[0], exp_model[1]

	numerator, denominator = 0, 0
	for [x, y] in data:
		if exp:
			numerator += (y - np.exp(w * x + b))**2
		else:
			numerator += (y - (w * x + b))**2
		denominator += (y - ybar)**2
	
	return 1 - numerator/denominator

print("QUESTION 5\nR^2 = ", compute_coef(S, exp_model))




# QUESTION 6: Model exp(w * sqrt(x) + b)
Ssecond = np.array([[np.sqrt(x), np.log(y)] for [x, y] in S])
final_model = lin_reg_1D(Ssecond)
print("QUESTION 6\nParameters of the final model : ", final_model)
#plot(Ssecond, final_model, "sqrt(years)", "ln(PCB)", "FinalModel_Ssecond")
plot(Sprime, final_model, "years", "ln(PCB)", "Q6_FinalModel_Sprime", test = "ln_final")
print('MSE of the final model : ', MSE_sqrt(S, final_model))

# compute the coefficient for the final model.
def compute_coef_final(data, exp_model):
	ybar = np.mean(data[:, 1])
	w, b = exp_model[0], exp_model[1]

	numerator, denominator = 0, 0
	for [x, y] in data:
		numerator += (y - (w * np.sqrt(x) + b))**2
		denominator += (y - ybar)**2
	
	return 1 - numerator/denominator

print("For exp model and Sprime R^2 = ", compute_coef(Sprime, exp_model, exp=False))
print("For final model and Sprime R^2 = ", compute_coef_final(Sprime, final_model))
