#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
os.getcwd()


# In[2]:


path='C:\\Users\\migla\\OneDrive\Desktop\\'


# In[3]:


data=pd.read_csv(path+"train.csv")
data_test=pd.read_csv(path+"test.csv")


# In[5]:


data.head()


# In[6]:


data.tail()


# In[8]:


data.shape


# In[9]:


data.duplicated().any()


# In[14]:


duplicated_columns=data.columns[data.T.duplicated()].tolist()


# In[15]:


(duplicated_columns)


# In[16]:


len(duplicated_columns)


# In[17]:


duplicated_columns=data.columns[data.T.duplicated()].tolist()
data = data.drop(duplicated_columns,axis=1)


# In[18]:


data.shape


# In[19]:


data.isnull().sum()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


sns.countplot(data['Activity'])
plt.xticks(rotation=35)
plt.show()


# In[22]:


X = data.drop('Activity',axis=1)
y= data['Activity']


# In[23]:


y


# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[25]:


y


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,
                                               random_state=42)


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[29]:


log  = LogisticRegression()
log.fit(X_train,y_train)


# In[30]:


y_pred1 = log.predict(X_test)
accuracy_score(y_test,y_pred1)


# In[31]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[28]:


y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[32]:


from sklearn.feature_selection import SelectKBest,f_classif


# In[33]:


k=200
selector = SelectKBest(f_classif,k=k)
X_train_selected = selector.fit_transform(X_train,y_train)
X_test_selected = selector.transform(X_test)


selected_indices=selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]
print(len(selected_features))


# In[34]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# In[35]:


estimator = RandomForestClassifier()


# In[36]:


k=101
rfe_selector = RFE(estimator,n_features_to_select=k)
X_train_selected_rfe = rfe_selector.fit_transform(X_train_selected,y_train)
X_test_selected_rfe = rfe_selector.transform(X_test_selected)

selected_indices_rfe = rfe_selector.get_support(indices=True)
selected_features_rfe = selected_features[selected_indices_rfe]
print(len(selected_features_rfe))


# In[34]:


print(len(selected_features_rfe))


# In[35]:


rf=RandomForestClassifier()
rf.fit(X_train_selected_rfe,y_train)


# In[36]:


y_pred_rf = rf.predict(X_test_selected_rfe)


# In[37]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred_rf)


# In[38]:


import joblib


# In[39]:


joblib.dump(rf,"model_rfe")


# In[40]:


joblib.dump(selector,"k_best_selector")


# In[41]:


joblib.dump(rfe_selector,"rfe_selector")


# In[42]:


data_test=data_test.drop("Activity",axis=1)


# In[43]:


duplicated_columns=data_test.columns[data_test.T.duplicated()].to_list()


# In[44]:


data_test=data_test.drop(duplicated_columns,axis=1)


# In[45]:


model=joblib.load('model_rfe')
selector=joblib.load('k_best_selector')
rfe_selector=joblib.load('rfe_selector')
selector=selector.transform(data_test)
X_test_selected_rfe = rfe_selector.transform(selector)


# In[54]:


model.predict(X_test_selected_rfe)


# In[ ]:





# In[4]:


import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        try:
            data = pd.read_csv(filepath)
            process_data(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")

def process_data(data):
    # Remove duplicated columns
    duplicated_columns = data.columns[data.T.duplicated()].tolist()
    data_test = data.drop(duplicated_columns, axis=1)

    model = joblib.load("model_rfe")
    selector = joblib.load('k_best_selector')
    rfe_selector = joblib.load('rfe_selector')

    # Transform the new data using the loaded SelectKBest object
    X_test_selected = selector.transform(data_test)

    # Transform the new data using the loaded RFE object
    X_test_selected_rfe = rfe_selector.transform(X_test_selected)

    # Make predictions
    y_pred = model.predict(X_test_selected_rfe)
    data_test['Predicted_Activity'] = y_pred

    save_file(data_test)

def save_file(data):
    savepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV Files", "*.csv")])
    if savepath:
        try:
            data.to_csv(savepath, index=False)
            messagebox.showinfo("Success", "File Saved Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

# Create a Tkinter GUI
root = tk.Tk()
root.title("Activity Classification")
root.geometry("200x200")

button1 = tk.Button(root, text="Open CSV File",
                    width=15,
                    height=2,
                    background="lightgreen",
                    activebackground="lightblue",
                    font=("Arial", 11, "bold"),
                    command=open_file)
button1.pack(pady=50)

root.mainloop()


# In[ ]:




