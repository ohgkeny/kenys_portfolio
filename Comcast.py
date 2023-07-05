#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


df = pd.read_csv("Comcast.csv")


# In[5]:


df.head(3)


# In[6]:


#Lets check the shape of our dataset
df.shape

#So we have 11 columns and 2224 records


# In[7]:


#lets check if there is any null or missing values
df.isnull().sum()

#as we check we dont have null or missing vals


# In[8]:


#lets check the types of variable we have
df.dtypes


# In[9]:


#lets get description about our data ex: descriptive of variables
df.describe(include='all')


# In[10]:


#Merging two columns into one date index [Date_month_year + Time]
df["date_index"] = df["Date_month_year"] + " " + df["Time"]


# In[11]:


df["date_index"]


# In[12]:


#lets convert the columns, from object to date type
df["date_index"] = pd.to_datetime(df["date_index"])
df["Date"] = pd.to_datetime(df["Date"])
df["Date_month_year"] = pd.to_datetime(df["Date_month_year"])


# In[13]:


# Setting index of dataset to date_index
df = df.set_index(df["date_index"])


# In[14]:


df.head(3)


# In[15]:


#lets check the name of all columns
df.columns


# In[16]:


# We will drop the ticket col as it doesn't seems useful in analysis 
df = df.drop(['Ticket #'], axis=1)


# In[17]:


df["Date_month_year"].value_counts()[:5]
#to see the max 5 counts


# In[18]:


#lets get the trend chart for the number of complaints at monthly and daily granularity levels 
df["Date_month_year"].value_counts().plot();


# In[19]:


# In the above graph we see a spark in complaints in 7th month 2015


# In[20]:


#lets plot monthly chart 
months = df.groupby(pd.Grouper(freq='M')).size().plot()
plt.xlabel('MONTHS')
plt.ylabel('FREQUENCY')
plt.title('MONTHLY TREND CHART')


# In[21]:


# from the above trend chart, we can clearly see that complaints for the month of June 2015 is maximum


# In[22]:


#lets try plotting daily chart 
df = df.sort_values(by='Date')
plt.figure(figsize=(6,6))
df['Date'].value_counts().plot()
plt.xlabel('DATE')
plt.ylabel('FREQUENCY')
plt.title('DAILY TREND CHART')


# In[23]:


df.Status.unique()

# Gives the unique values for status column


# In[24]:


#lets create a new categorical variable with value as Open & Closed. Open & Pending is to be categorised as Open & Closed & Solved is to be categorized as Closed 
df["NewStatus"] = ["Open" if (status=="Open" or status=="Pending") else "Close" for status in df["Status"]]


# In[25]:


df.head(5)


# In[26]:


#lets create a table with the frequency of complaint types
df['Customer Complaint'].value_counts(dropna=False)[:9]


# In[27]:


df['Customer Complaint'].value_counts(dropna=False)[:9].plot.barh()


# In[28]:


###From the above chart we see that Internet has high count of word after comcast


# In[29]:


df.groupby(["State"]).size().sort_values(ascending=False).to_frame().reset_index().rename({0: "Count"}, axis=1)[:5]


# In[30]:


Staus_complaints = df.groupby(["State", "NewStatus"]).size().unstack().fillna(0)


# In[31]:


Staus_complaints


# In[32]:


Staus_complaints["Open"].sort_values(ascending=False)
#These are the state with counts of open tickets


# In[33]:


Staus_complaints.plot(kind="barh", figsize=(30,50), stacked=True)
plt.rcParams.update({"font.size": 20})
# This is a plot for open and close cases for all states


# In[34]:


#lets check which state has the maximum complaints
df.groupby(["State"]).size().sort_values(ascending=False)[:5]


# In[35]:


#As we see Georgia has the maximum complaints


# In[36]:


Status_complaints = df.groupby(["State", "NewStatus"]).size().unstack()
print(Status_complaints)


# In[37]:


unresolved_data = df.groupby(["State","NewStatus"]).size().unstack().fillna(0).sort_values(by="Open", ascending=False)


# In[38]:


unresolved_data['Unresolved_cmp_prct'] = unresolved_data["Open"]/unresolved_data["Open"].sum()*100


# In[39]:


print(unresolved_data)


# In[40]:


unresolved_data.plot()


# In[41]:


# From the above details we see that Georgia has the maximum resolved complaints


# In[42]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[43]:


txt = df["Customer Complaint"].values


# In[44]:


wc = WordCloud(width=200, height=100, background_color="black", stopwords=STOPWORDS).generate(str(txt))


# In[45]:


fig = plt.figure(figsize=(20,20), facecolor='k', edgecolor='w')
plt.imshow(wc, interpolation="bilinear") #bicubic and there are other interpolation method as well
plt.axis("off")
plt.tight_layout()
plt.show()


# In[46]:


# Document term matrix
# Doc list = ["Comcast cable internet speed",...]
# Doc Index = [0,1,2,3]
# Doc term matrix = [(0,1),(1,1)]....

# LDA
# Latent - Hidden
# Dirichlet - Kind of probability distribution


# In[47]:


import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer


# In[48]:


nltk.download('stopwords')
nltk.download('wordnet')


# In[49]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# In[50]:


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join([ch for ch in stop_free if ch not in exclude])
    normalised = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalised


# In[51]:


doc_complete = df["Customer Complaint"].tolist()


# In[52]:


# nltk.download('omw-1.4')
doc_clean = [clean(doc).split() for doc in doc_complete]


# In[53]:


import gensim
from gensim import corpora


# In[54]:


dictionary = corpora.Dictionary(doc_clean)
print(dictionary)


# In[55]:


doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# In[56]:


doc_term_matrix


# In[57]:


from gensim.models import LdaModel


# In[58]:


Num_topics = 9


# In[59]:


ldamodel = LdaModel(doc_term_matrix, Num_topics, id2word=dictionary, passes=30)


# In[60]:


topics = ldamodel.show_topics()


# In[61]:


for topic in topics:
    print(topic)
    print()


# In[62]:


word_dict = {}
for i in range(Num_topics):
    words = ldamodel.show_topic(i, topn=20)
    word_dict["Topic # " + "{}".format(i)] = [i[0] for i in words]


# In[63]:


word_dict_ = pd.DataFrame(word_dict)


# In[64]:


word_dict_


# In[65]:


ldamodel.show_topic(0, topn=20)


# In[ ]:


get_ipython().system('pip install pyLDAvis --user')


# In[ ]:


import pyLDAvis
import pyLDAvis.gensim as gensimvis
pyLDAvis.enable_notebook()


# In[ ]:


Lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False)
pyLDAvis.display(Lda_display)

