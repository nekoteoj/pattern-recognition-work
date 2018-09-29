import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
#from sympy.solvers import solve
#from sympy import *
import math
data=pd.read_csv("hr-employee-attrition-with-null.csv")
#print(data.head())
#print(data.describe())
plt.ion()
#Cleaning data
data.loc[data["Attrition"] == "No", "Attrition"] = 0.0
data.loc[data["Attrition"] == "Yes", "Attrition"] = 1.0
cat_cols = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "Over18", "OverTime"]
#Education, EnvironmentSatisfaction, JobInvolvement, JobLevel, JobSatisfaction, PerformanceRating, RelationshipSatisfaction, StockOptionLevel and WorkLifeBalance, are already categorized???
for col in cat_cols:
	data[col] = pd.Categorical(data[col]).codes
data = data.drop(columns = "EmployeeNumber")
data = data.drop(columns = "Unnamed: 0")

def train_test_split(df, label, percentage=0.9):
	train_idx = np.zeros(len(label), dtype=bool)
	la = np.array(label)
	uq = np.unique(la)
	for val in uq:
		val_idx = np.nonzero(la==val)[0]
		np.random.shuffle(val_idx)
		count = int(percentage*len(val_idx))
		use_idx = val_idx[:count]
		train_idx[use_idx] = True
	train = data[train_idx]
	test = data[~train_idx]
	return train, test


train, test = train_test_split(data, data["Attrition"])
all_col = list(train)
train_mode = train["Attrition"].mode()[0]
#get rid of nan
#for col in all_col:
	#train[col] = train[~np.isnan(train[col])]


for col in all_col:
	print("Generating histogram for ", col, " class...")
	input_df = train[~np.isnan(train[col])]
	input_df = input_df[col]
	#print(input_df)
	hist, bin_edge = np.histogram(input_df, 40)
	print(hist)
	zero_bin = np.count_nonzero(hist==0)
	print("There are ", zero_bin, " bins which has 0 element in it")
	#plt.fill_between(bin_edge.repeat(2)[1:-1],hist.repeat(2),facecolor='steelblue')
	#plt.show()
	#plt.pause(0.1)
	#plt.close()

"""
T4: We can use gaussian distribution to estimate some of the features. By inspecting the histogram, we can observe that Age might be the only feature that folow single gaussian distribution. Another categorical features, such as Education and Gender cannot be used since there are a lot of bins with zero elements. Also, some of the continuous features, such as MonthlyRate, even though there is no bins with zero elements but the histogram does not folow gaussian distribution. However, we might still be able to use GMM to estimate those features.
"""

"""
T5: From the histogram of all features, we can see that a lot of features that is categorical in nature, such as BusinessTravel, Department , Education etc., have a lot of bins with zero elements count. Those features cannot be mapped to any probabilistic distribution and so is not a good discretization.
"""

def hist_plot(data, feature, bin_c):
	print("Generating histogram for ", feature, " class with ", bin_c, " bins...")
	input_df = data[~np.isnan(data[feature])]
	pl_value = input_df[feature]
	bin = np.linspace(pl_value.min(), pl_value.max(), bin_c+1)
	bin_sep = np.digitize(pl_value, bin)
	hist = np.bincount(bin_sep)
	print("The bin edge is...")
	print(bin)
	print("The histogram is...")
	print(hist)
	#plt.fill_between(bin.repeat(2)[1:-1],hist.repeat(2),facecolor='steelblue')
	#plt.show()
	#plt.pause(3)
	#plt.close()
print("="*100)
test_bin = ["Age", "MonthlyIncome", "DistanceFromHome"]
for col in test_bin:
	for bin in [10, 40, 100]:
		hist_plot(train, col, bin)

def con_bin_likely(data, feature, bin=10):
	input_df = data[~np.isnan(data[feature])]
	pl_value = input_df[feature]
	total = len(pl_value)
	bin_edge = np.linspace(pl_value.min(), pl_value.max(), bin+1)
	bin_sep = np.digitize(pl_value, bin_edge)
	bin_count = np.bincount(bin_sep)
	likely = bin_count/total
	return likely, bin_edge

def cat_bin_likely(data, feature):
	input_df = data[~np.isnan(data[feature])]
	pl_value = input_df[feature]
	cat_num = len(np.unique(pl_value))
	total = len(pl_value)
	bin_edge = np.linspace(pl_value.min()-0.5, pl_value.max()+0.5, cat_num+1)
	bin_sep = np.digitize(pl_value, bin_edge)
	bin_count = np.bincount(bin_sep)
	likely = bin_count/total
	return likely, bin_edge

category_features = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'Over18', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']
continuous_features = ['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeCount', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'StandardHours', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
all_features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

def hist_store(bin=10):
	leave = {}
	stay = {}
	df_leave = train[train["Attrition"]==1]
	df_stay = train[train["Attrition"]==0]
	for col in category_features:
		likely_leave, bin_edge_leave = cat_bin_likely(df_leave, col)
		leave[col] = [likely_leave, bin_edge_leave]
		likely_stay, bin_edge_stay = cat_bin_likely(df_stay, col)
		stay[col] = [likely_stay, bin_edge_stay]
	for col in continuous_features:
		likely_leave, bin_edge_leave = con_bin_likely(df_leave, col, bin)
		leave[col] = [likely_leave, bin_edge_leave]
		likely_stay, bin_edge_stay = con_bin_likely(df_stay, col, bin)
		stay[col] = [likely_stay, bin_edge_stay]
	return leave, stay

def normal_store():
	leave = {}
	stay = {}
	df_leave = train[train["Attrition"]==1]
	df_stay = train[train["Attrition"]==0]
	for col in category_features:
		mean_leave = df_leave[col].mean()
		standard_div_leave = df_leave[col].std()
		if standard_div_leave == 0.0:
			standard_div_leave = 1e-10
		leave[col] = [mean_leave, standard_div_leave]
		mean_stay = df_stay[col].mean()
		standard_div_stay = df_stay[col].std()
		if standard_div_stay == 0.0:
			standard_div_stay = 1e-10
		stay[col] = [mean_stay, standard_div_stay]
	for col in continuous_features:
		mean_leave = df_leave[col].mean()
		standard_div_leave = df_leave[col].std()
		if standard_div_leave == 0.0:
			standard_div_leave = 1e-10
		leave[col] = [mean_leave, standard_div_leave]
		mean_stay = df_stay[col].mean()
		standard_div_stay = df_stay[col].std()
		if standard_div_stay == 0.0:
			standard_div_stay = 1e-10
		stay[col] = [mean_stay, standard_div_stay]
	return leave, stay

hist_leave, hist_stay = hist_store(10)
normal_leave, normal_stay = normal_store()



def p_hist_leave(x, feature):
	hist, bin_edge = hist_leave[feature]
	index = int(np.digitize(x, bin_edge))
	#print(hist)
	#print(bin_edge)
	#print(x)
	#print(feature)
	return hist[index]

def p_hist_stay(x, feature):
	hist, bin_edge = hist_stay[feature]
	index = int(np.digitize(x, bin_edge))
	#print(hist)
	#print(bin_edge)
	#print(x)
	#print(feature)
	return hist[index]

def evaluation(predicted, label, printed=True):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	num = len(label)
	for i in range(num):
		if predicted[i] == 1 and label[i] == 1:
			tp += 1
		elif predicted[i] == 0 and label[i] == 0:
			tn += 1
		elif predicted[i] == 1 and label[i] == 0:
			fp += 1
		elif predicted[i] == 0 and label[i] == 1:
			fn += 1
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	try:
		precision = tp/(tp+fp)
	except ZeroDivisionError:
		precision = -1
	try:
		recall = tp/(tp+fn)
	except ZeroDivisionError:
		recall = -1
	try:
		f1 = 2*(recall*precision)/(recall+precision)
	except ZeroDivisionError:
		f1 = -1
	if printed == True:
		print("Evaluation result:")
		print("True possitive:", tp)
		print("True negative:", tn)
		print("False possitive:", fp)
		print("False negative:", fn)
		print("Accuracy:", accuracy)
		print("Precision:", precision)
		print("Recall:", recall)
		print("F1 score:", f1)
	else:
		return tp, tn, fp, fn, accuracy, precision, recall, f1

prior_leave = len(train[train["Attrition"]==1])/len(train)
prior_stay = len(train[train["Attrition"]==0])/len(train)
true_label = np.array(test["Attrition"])
test = test.drop(columns = "Attrition")
test_features = list(test)
num_features = len(test_features)
num_test = len(test)
test_data = np.array(test)

def histogram_model(threshold=0):
	predicted = np.zeros(num_test)
	for i in range(num_test):
		considered_row = test_data[i]
		log_likely = np.log(prior_leave)-np.log(prior_stay)
		for j in range(num_features):
			considered_feature = test_features[j]
			if np.isnan(considered_row[j]):
				continue
			log_likely += np.log(p_hist_leave(considered_row[j], considered_feature))
			log_likely -= np.log(p_hist_stay(considered_row[j], considered_feature))
		if log_likely > threshold:
			predicted[i] = 1
		else:
			predicted[i] = 0
	return predicted


def normal_model(threshold=0):
	predicted = np.zeros(num_test)
	log_likely = np.full((num_test),np.log(prior_leave)-np.log(prior_stay))
	for j in range(num_features):
		considered_feature = test_features[j]
		col_data = test_data[::, j]
		mean_leave, std_leave = normal_leave[considered_feature]
		mean_stay, std_stay = normal_stay[considered_feature]
		temp_arr = np.log(norm(mean_leave, std_leave).pdf(col_data))
		ma_array = np.ma.array(temp_arr, mask=np.isnan(temp_arr))
		log_likely = log_likely+ma_array
		temp_arr = np.log(norm(mean_stay, std_stay).pdf(col_data))
		ma_array = np.ma.array(temp_arr, mask=np.isnan(temp_arr))
		log_likely = log_likely-ma_array
		if np.ma.is_masked(log_likely):
			log_likely = log_likely.data
	predicted = log_likely > threshold
	return predicted



predicted_hist = histogram_model(0)
print("Histogram model")
evaluation(predicted_hist, true_label)
predicted_norm = normal_model(0)
print("Gaussian model")
evaluation(predicted_norm, true_label)


predicted_random = np.random.randint(0,2, size=num_test)
print("Random model")
evaluation(predicted_random, true_label)

predicted_mode = np.array([train_mode]*num_test)
print("Majority model")
evaluation(predicted_mode, true_label)



t = np.arange(-5,5,0.05)
print("threshold finding...")
print("Histogram model...")
plt.title("ROC for discretization bin = 10")
for s in t:
	predicted_threshold = histogram_model(s)
	tp, tn, fp, fn, accuracy, precision, recall, f1 = evaluation(predicted_threshold, true_label, False)
	plt.plot(fp, recall)
	print("using threshold={:.2f} accuracy={:.3f} precision={:.3f} recall={:.3f} f1={:.3f}".format(s, accuracy, precision, recall, f1))
plt.show()
plt.savefig("bin10")
plt.pause(1)
plt.close()

plt.title("ROC for normal model")
print("Normal model...")
for s in t:
	predicted_threshold = normal_model(s)
	tp, tn, fp, fn, accuracy, precision, recall, f1 = evaluation(predicted_threshold, true_label, False)
	plt.plot(fp, recall)
	print("using threshold={:.2f} accuracy={:.3f} precision={:.3f} recall={:.3f} f1={:.3f}".format(s, accuracy, precision, recall, f1))
plt.show()
plt.savefig("normal")
plt.pause(1)
plt.close()

print("Changing number of discretization bin to 5...")
hist_leave, hist_stay = hist_store(5)

plt.title("ROC for discretization bin = 5")
print("Histogram model...")
for s in t:
	predicted_threshold = histogram_model(s)
	tp, tn, fp, fn, accuracy, precision, recall, f1 = evaluation(predicted_threshold, true_label, False)
	plt.plot(fp, recall)
	print("using threshold={:.2f} accuracy={:.3f} precision={:.3f} recall={:.3f} f1={:.3f}".format(s, accuracy, precision, recall, f1))
plt.show()
plt.savefig("bin5")
plt.pause(1)
plt.close()
