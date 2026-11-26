import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
df = pd.read_csv("D:\Erynta_10395\code\Big-Data-Analytics\data\employee_data.csv") 
print(f"Shape: {df.shape}")           # (600, 15) 
print(df.isnull().sum())              # All zeros — no missing values 
print(df.duplicated().sum())          # 0 duplicate rows 

# 1. Parse hire dates and compute tenure 
df["HireDate"] = pd.to_datetime(df["HireDate"]) 
df["TenureYears"] = (pd.Timestamp("2024-01-01") - 
df["HireDate"]).dt.days / 365.25 
# 2. Clip salary outliers using IQR 
Q1, Q3 = df["Salary"].quantile([0.25, 0.75]) 
IQR = Q3 - Q1 
df["Salary"] = df["Salary"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR) 
# 3. Ordinal encode Education 
edu_order = {"High School":1, "Associate":2, "Bachelor":3, 
"Master":4, "PhD":5} 
df["EducationLevel"] = df["Education"].map(edu_order) 

print(f"Salary IQR bounds: [{Q1 - 1.5*IQR:,.0f}, {Q3 + 1.5*IQR:,.0f}]")
print("New column TenureYears computed from HireDate")
print("New column EducationLevel with numeric ordinal values (1-5)")
print(df[["HireDate", "TenureYears", "Salary", "Education", "EducationLevel"]].head())

# Average salary by department 
dept_salary = df.groupby("Department")["Salary"].mean().sort_values(ascending=False
) 
# Performance by education 
edu_perf = df.groupby("Education")["PerformanceScore"].mean() 
# Correlation matrix 
corr = df[["Age","Salary","PerformanceScore","YearsExperience","TrainingHours"]].corr() 


print(f"{dept_salary.round(2)} \n")
print(f"{edu_perf} \n")
print(f"{corr} \n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Average Salary by Department (เรียงจากน้อยไปมากเพื่อให้ค่าสูงสุดอยู่บนสุดเหมือนกราฟแนวนอนในโจทย์)
dept_salary_plot = dept_salary.sort_values(ascending=True)
axes[0, 0].barh(dept_salary_plot.index, dept_salary_plot.values)
axes[0, 0].set_title("Average Salary by Department")
axes[0, 0].set_xlabel("Salary (USD)")

# Top-right: Distribution of Performance Scores
axes[0, 1].hist(df["PerformanceScore"], bins=15, edgecolor="black")
axes[0, 1].set_title("Distribution of Performance Scores")
axes[0, 1].set_xlabel("Performance Score")
axes[0, 1].set_ylabel("Frequency")

# Bottom-left: Salary vs Years of Experience
axes[1, 0].scatter(df["YearsExperience"], df["Salary"], alpha=0.6)
axes[1, 0].set_title("Salary vs Years of Experience")
axes[1, 0].set_xlabel("Years of Experience")
axes[1, 0].set_ylabel("Salary (USD)")

# Bottom-right: Employee Status Distribution
status_counts = df["Status"].value_counts()
axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%")
axes[1, 1].set_title("Employee Status Distribution")

plt.suptitle("Lab 1 — Employee Data Analytics Dashboard", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("lab1_dashboard.png", dpi=150)
plt.show()