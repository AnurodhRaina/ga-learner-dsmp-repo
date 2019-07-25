# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data_file=path
data=np.genfromtxt(data_file, delimiter=",", skip_header=1)

print("\nData: \n\n", data)

print("\nType of data: \n\n", type(data))
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
census=np.concatenate((data,new_record))
#Code starts here



# --------------
#Code starts here
age=census[:,0]
max_age=np.max(age)
min_age=np.min(age)
age_mean=age.mean()
age_std=np.std(age)
print(max_age,min_age,age_mean,age_std)


# --------------
#Code starts here
new=census[:,2]
new=new.astype('int32')
race_0=census[census[:,2]==0]
#race_0=new[new==0]
race_1=census[census[:,2]==1]
race_2=census[census[:,2]==2]
race_3=census[census[:,2]==3]
race_4=census[census[:,2]==4]
print(race_0)#,race_1,race_2,race_3,race_4)
len_0=len(race_0)
len_1=len(race_1)
len_2=len(race_2)
len_3=len(race_3)
len_4=len(race_4)
length=np.array([len_0,len_1,len_2,len_3,len_4])
minority_race= list(length).index(length.min())
print(length)
print(minority_race)



# --------------
#Code starts here
senior_citizens=census[census[:,0]>60]
working_hours_sum=senior_citizens[:,6].sum()
senior_citizens_len=len(senior_citizens)
avg_working_hours=working_hours_sum/senior_citizens_len

print(avg_working_hours)


# --------------
#Code starts here
high=census[census[:,1]>10]
low=census[census[:,1]<=10]

avg_pay_high=high[:,7].mean()
avg_pay_low=low[:,7].mean()

print(avg_pay_high,avg_pay_low)


