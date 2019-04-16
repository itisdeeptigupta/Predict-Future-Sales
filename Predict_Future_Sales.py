
# coding: utf-8

# In[2]:


import numpy as np
from numpy.random import seed
from numpy.random import randn

import pandas as pd

from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

import seaborn as sns
sns.set(style="whitegrid")

import math
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import csv
import time


# #### Load the data sales train data.

# In[6]:


items = pd.read_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\items.csv")
item_cat = pd.read_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\item_categories.csv")

sales_train = pd.read_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\sales_train_v2.csv")
shops = pd.read_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\shops.csv")

sales_test = pd.read_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\test.csv")
sample_submission = pd.read_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\sample_submission.csv")


# #### Data Cleansing and staging

# In[7]:


sales_train.head(10)


# In[8]:


sales_train.info()


# #### Check if the train and test data contain any missing values
# 

# In[9]:


sales_train[sales_train.isnull().any(axis=1)].head()     #NO MISSING VALUE


# In[10]:


sales_test[sales_test.isnull().any(axis=1)].head()   #NO MISSING VALUE


# #### Check for presence of any outliers in the data
# 

# In[11]:


plt.plot(sales_train['item_id'], sales_train['item_price'], 'o', color='blue');


# In[12]:


sales_train[sales_train.item_price > 250000]


# In[13]:


items[items.item_id == 6066]


# In[14]:


item_cat[item_cat.item_category_id == 75]


# In[15]:


shops[shops.shop_id == 12]


# #### Conclusion: The record which first appeared to be an outlier seems to be a genuine sale
# 

# In[16]:


# staging for further analysis
sales_train_sub = sales_train
sales_train_sub['date'] =  pd.to_datetime(sales_train_sub['date'],format= '%d.%m.%Y')
sales_train_sub['month'] = pd.DatetimeIndex(sales_train_sub['date']).month
sales_train_sub['year'] = pd.DatetimeIndex(sales_train_sub['date']).year
sales_train_sub = sales_train_sub.iloc[:,1:8]
sales_train_sub.head(10)


# #### Tableau Report

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1554949555004' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;sh&#47;shop_items_count-itemFilter&#47;Sheet3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='shop_items_count-itemFilter&#47;Sheet3' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;sh&#47;shop_items_count-itemFilter&#47;Sheet3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1554949555004');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# 
# ### Approach 1: Decision Tree Classification

# In[17]:


monthly_sales=sales_train_sub.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg({"item_cnt_day":"sum"})

monthly_sales['date_block_num'] = monthly_sales.index.get_level_values('date_block_num') 
monthly_sales['shop_id'] = monthly_sales.index.get_level_values('shop_id') 
monthly_sales['item_id'] = monthly_sales.index.get_level_values('item_id') 
monthly_sales.reset_index(drop=True, inplace=True)

monthly_sales = monthly_sales.reindex_axis(['date_block_num','shop_id','item_id','item_cnt_day'], axis=1)
monthly_sales.head(10)


# In[18]:


from sklearn import tree
model_DTC = tree.DecisionTreeClassifier(criterion='gini') 

cols = ['shop_id','date_block_num','item_id']
X = monthly_sales[cols] # Predictor columns
y = monthly_sales.item_cnt_day  # Target variable


# In[20]:


model_DTC.fit(X, y)


# In[21]:


X_test_DTC = sales_test[['shop_id','item_id']]
X_test_DTC.insert(loc=1, column='date_block_num', value='34')  
X_test_DTC.head(10)


# Prediction using decision tree classifier

# In[22]:


predicted_raw_DTC= model_DTC.predict(X_test_DTC)

predicted_DTC = pd.DataFrame(predicted_raw_DTC)
predicted_DTC = predicted_DTC.join(X_test_DTC)

predicted_DTC.columns  = ['item_cnt', 'shop_id', 'date_block_num','item_id']
predicted_DTC = predicted_DTC.reindex_axis(['shop_id','date_block_num','item_id','item_cnt'], axis=1)

print(predicted_DTC.head(10))

predicted_DTC.to_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\SalesTest_DecisionTreeClass.csv", sep=',')


# ### Approach 2: Decision Tree Regressor

# In[23]:


X2 = monthly_sales[cols] # Predictor columns
Y2 = monthly_sales.item_cnt_day  # Target variable

# Fitting Simple Linear Regression model to the data set
from sklearn.tree import DecisionTreeRegressor
model_DTR = DecisionTreeRegressor(random_state = 0)
model_DTR.fit(X2, Y2)


# In[24]:


X2_test_DTR = sales_test[['shop_id','item_id']]
X2_test_DTR.insert(loc=1, column='date_block_num', value='34')


predicted_raw_DTR = pd.DataFrame(model_DTR.predict(X2_test_DTR))
predicted_raw_DTR = X2_test_DTR.join(predicted_raw_DTR)

predicted_raw_DTR.columns  = ['shop_id', 'date_block_num','item_id', 'item_cnt']
predicted_DTR = predicted_raw_DTR.reindex_axis(['shop_id','date_block_num','item_id','item_cnt'], axis=1)

print(predicted_DTR.head(10))

predicted_DTR.to_csv("C:\\Study\\530-MachineLearningI\\Project\\predict_future_sales\\SalesTest_DecisionTreeRegr.csv", sep=',')


# In[25]:


#visualizing decision tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(model_DTR, out_file=dot_data, max_depth=5,
filled=True, rounded=True,
special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# # APPROACH 3: Designing Linear Regression Model

# ### APPROACH
# 
# #### 1. A distinctive model will be designed and trained for each combination of shopid_itemid 
# #### 2. Sales_test data will be fed in the same format (shopid_itemid) to that particular model
# #### 3. Item_Count for the 34th month will be predicted for that shop and item id combination
# 
# #### E.g. for shopid = 5, itemid = 4872, a model corresponding to combination 5_4872 will be trained and next month's item count will be predicted using the same model

# In[26]:


df = pd.DataFrame(sales_train,columns=['date','date_block_num','shop_id','item_id','item_price','item_cnt_day'])


# Taking shop 5 alone because of huge data and now, we are going to subset the item count for all months in this shop

# In[27]:


df=pd.DataFrame(df.loc[df['shop_id'] == 5])


# Get the unique shop_id's to subset. If we dont do the filter in the above step, below code will add all the shops in a list. 
# For now, the list will only have 5

# In[28]:


shops=df['shop_id'].unique().tolist()
print(shops)


# Creating a dictionary "shopdf" to refer to different shops subset of data, in our case it will show the data corresponding to shop 5

# In[29]:


shopdf={}
for shop in shops:
    shopdf[shop]=pd.DataFrame(df.loc[df['shop_id'] ==shop])
print(shopdf)


# Created a dictionary "shopdf_agg" to refer to the aggreagated monthly count data of the items sold for shop 5

# In[30]:


shopdf_agg={}
for shop in shops:
    s=shopdf[shop].groupby(['date_block_num','item_id'])['item_cnt_day'].agg({'item_cnt_agg':'sum'})
    s['date_block_num'] = s.index.get_level_values('date_block_num') 
    s['item_id'] = s.index.get_level_values('item_id') 
    s.reset_index(drop=True, inplace=True)
    s = s.reindex_axis(['date_block_num','item_id','item_cnt_agg'], axis=1)
    shopdf_agg[shop]=s
print(shopdf_agg)


# #### data_ideal_df is an ideal dataframe for all the months (0-33), in case a shop has not sold any item in kth month, we will merge it with data_ideal_df to add the count as '0'
# 
# #### Created dictionary "shop_item" to list all the items in the shop and dictionary sh_itm_mn to subset the items in the given shop

# In[31]:


data_ideal_df = [[0, 0], [1, 0],[2, 0],[3, 0],[4, 0],[5, 0],[6, 0],[7, 0],[8, 0],
                 [9, 0],[10, 0],[11, 0],[12, 0],[13, 0],[14, 0],[15, 0],[16, 0],[17, 0],[18, 0],
                 [19, 0],[20, 0],[21, 0],[22, 0],[23, 0],[24, 0],[25, 0],[26, 0],[27, 0],[28, 0],
                 [29, 0],[30, 0],[31, 0],[32, 0],[33, 0]] 
ideal_df = pd.DataFrame(data_ideal_df, columns = ['date_block_num', 'item_id'])

shop_item={}
sh_itm_mn={}
model=[]#initialize list of unique combination of item in the shop which will be later used to refer to the model
for shop in shops:
    shop_item[shop]=shopdf_agg[shop]['item_id'].unique().tolist()# get the list of item ids in the shop
    for item in shop_item[shop]:
        Title='Shop_'+str(shop)+' - Item_'+str(item)
        print(Title)
        #print(shopdf_agg[shop][item])
        mod=str(shop)+'_'+str(item)#unique identifier for item in the shop
        print(mod)
        model.append(mod)
        temp = pd.DataFrame(shopdf_agg[shop].loc[shopdf_agg[shop]['item_id'] ==item]) #subset items in the shop
        ideal_df['item_id'] = item  
        
        # merging the two dataframes so that every item contains every dateblocknum even if it is zero
        sh_itm_mn[mod]= pd.merge(temp, ideal_df, on=['date_block_num', 'item_id'], how='outer', sort= True).fillna(0)
        print(sh_itm_mn[mod])


# #### Checking the assumptions for linear regression
# 1. Normality (Shapiro Wilk Test)
# 2. Multi-Collinearity (Correlation Matrix)
# 3. Homoscedastity (Levene's Test)
# 4. Linearity (QQ Plots)
#  

# In[32]:


print("Shapiro-Wilk Test for Normality. \n")
for i in model:
    shapiro_p_value = round((stats.shapiro(sh_itm_mn[i]['item_cnt_agg'])[0]),2)
    print("For shop_item {}, the p-value is".format(i) , shapiro_p_value )
    
    if shapiro_p_value > 0.05:
        print('Sample looks Gaussian (fail to reject H0). \n')
    else:
        print('Sample does not look Gaussian (reject H0). \n')

#Every model has a p-value greater than 0.05 which suggests that the data is normally distributed


# In[ ]:


print("Check Correlation between variables to detect Multi-Collinearity. \n")

for i in model:
    
    df_cor = sh_itm_mn[i].corr()
    pd.DataFrame(np.linalg.inv(df_cor.values), index = df_cor.index, columns=df_cor.columns)
    print("For shop_item {}, the correlation matrix - ".format(i),'\n', df_cor,'\n\n')

# Since the absolute correlation coefficient <= 0.95, we can suggest that they are not significant 
# and multi-collinearity does not exists


# In[ ]:


## Homoscedasticity

print("Finding homogenous variances with Levene's Test. \n")

for i in model:
    
    levenes_p_value = round(stats.levene(sh_itm_mn[i].item_id, sh_itm_mn[i].item_cnt_agg)[1],4)
    print("For shop_item {}, the p-value is".format(i),levenes_p_value, '\n')
   

# The test reveals a p-value greater than 0.05, indicating that there is no significant 
# difference in variances between the groups in location.


# In[ ]:


# QQ Plot for Linearity

seed(123)
# q-q plot

print("Displaying Linearity with QQ plots. \n")
for i in model:
    
    print("For shop_item {}".format(i), '\n')
    qqplot(sh_itm_mn[i].item_cnt_agg, line='s') 
    plt.show()


# In[ ]:


print("Displaying Linearity with QQ plots for Shop 5 and Item_ID 1830. \n")
    
print("For shop_item 5_1830".format('5_1830'), '\n')
qqplot(sh_itm_mn['5_1830'].item_cnt_agg, line='s') 
plt.show()


# In[ ]:


print("Displaying Linearity with QQ plots for Shop 5 and Item_ID 1905. \n")
    
print("For shop_item 5_1905".format('5_1905'), '\n')
qqplot(sh_itm_mn['5_1905'].item_cnt_agg, line='s') 
plt.show()


# #### A linear regression model is created for each shop_item combination
# 

# In[ ]:


# with statsmodels
shop_item_statsmodel={}#initialize dictionary model to store shop-items specific model using statsmodel
for i in model:
    print("\n")
    print(i)
    X=sh_itm_mn[i]['date_block_num']
    Y=sh_itm_mn[i]['item_cnt_agg']
    X = sm.add_constant(X) # adding a constant
    shop_item_statsmodel[i] = sm.OLS(Y, X).fit()
    predictions = shop_item_statsmodel[i].predict(X) 
    print_model = shop_item_statsmodel[i].summary()
    print(print_model)


# Check the prediction from stats model

# In[16]:


test_mod=[]
monthToPredict=34
t=int(time.time())
filename='salesPrediction'+str(t)+'.csv'
with open(filename,'a+') as csvFile:
    writer = csv.writer(csvFile)
    header=['Shop_ID','Item_ID','Item_Cnt']
    writer.writerow(header)
    for index, row in sales_test.head(30).iterrows():
        shop_test=row['shop_id']
        item_test=row['item_id']
        mod=str(shop_test)+'_'+str(item_test)
        test_mod.append(mod)
        if mod not in model:
            print('shop_item: {} is not in training set'.format(mod))
            row=[shop_test,item_test,'NA']
        else:
            predict=[]
            for i in range(35):
                month=pd.DataFrame([[i]],columns=['date_block'])
                X = sm.add_constant(month)
                raw_pred=shop_item_statsmodel[mod].predict(X)
                if raw_pred < 0:
                    pred=0
                else:
                    pred=math.ceil(raw_pred)
                predict.append(pred)
                if i==monthToPredict:
                    predictedValue=pred
            row=[shop_test,item_test,predictedValue]
            plt.figure()
            plt.scatter(sh_itm_mn[mod]['date_block_num'], sh_itm_mn[mod]['item_cnt_agg']) 
            graphTitle='Shop_Item:'+str(mod)+'  Value Predicted for month block '+ str(monthToPredict)+': '+str(predictedValue)
            plt.title(graphTitle, fontsize=14) 
            plt.xlabel('month block', fontsize=14) 
            plt.ylabel('item count', fontsize=14) 
            plt.grid(True) 
            plt.plot(predict,color='red', linewidth=3)
            plt.show()
        writer.writerow(row)
        #print(row)
csvFile.close()            

