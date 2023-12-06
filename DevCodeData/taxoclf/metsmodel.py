# usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay,f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np
from utils import *
import umap.umap_ as umap
from sklearn.manifold import TSNE

hmdb_mets_df = pd.read_csv('../output/hmdb_mets_df.csv')
hmdb_mets_df.head()

hmdb_mets_df['md'] = hmdb_mets_df['md'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))

le = LabelEncoder()
le.fit_transform(hmdb_mets_df['label'])

X = np.asarray([np.asarray(i) for i in hmdb_mets_df['md'].values])
y = le.fit_transform(hmdb_mets_df['label'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle=True,stratify=y)

RF = RandomForestClassifier()
SVM = svm.SVC(decision_function_shape='ovo')
KNN = KNeighborsClassifier(n_neighbors=5)
SGD = SGDClassifier(random_state=42)
XGB = GradientBoostingClassifier(random_state=42)

model_dict = {'rf':RF,
            'svm':SVM,
            'knn':KNN,
            'sgd':SGD,
            'xgb':XGB}

def model_eval(label, pred):
    accuracy = accuracy_score(label, pred)
    print("Accuracy: %.3f" % accuracy)
    precision = precision_score(label, pred, average='macro')
    print("Precision: %.3f" % precision)
    recall = recall_score(label, pred, average='macro')
    print("Recall: %.3f" % recall)
    f1 = f1_score(label, pred, average='macro')
    print("f1_score: %.3f" % f1)

model_list = list(model_dict.keys())

for j in range(len(model_list)):
    key = model_list[j]
    print(key)
    model = model_dict[key]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_result = model_eval(y_test,y_pred)
    print('----------------------')

# from descriptastorus.descriptors import rdNormalizedDescriptors

# generator = rdNormalizedDescriptors.RDKit2DNormalized()
# feature_name = [i[0] for i in generator.columns] # list of tuples:  (descriptor_name, numpytype) ...

# RF.fit(X_train, y_train)
# sorted_idx = RF.feature_importances_.argsort()
# feature_name = [feature_name[i] for i in sorted_idx]
# sns.set_style('ticks')
# plt.figure(figsize=(10,10))
# plt.rcParams['font.size'] = '16'
# ax = sns.barplot(y=feature_name[-20:],x=RF.feature_importances_[sorted_idx[-20:]], palette="crest")
# ax.invert_yaxis()
# plt.xlabel("Feature importance",fontsize=20)
# plt.ylabel("Molecular descriptor",fontsize=20)
# plt.savefig('../output/mets_feature_importance.png',dpi=600,bbox_inches='tight')


'''
t = list(le.inverse_transform([0,1,2,3,4,5,6,7,8,9]))

t_dict = {'Alkaloids and derivatives':'AKA', 
        'Benzenoids':'BZO',
        'Nucleosides, nucleotides, and analogues':'NCL',
        'Organic acids and derivatives':'OAC',
        'Organic nitrogen compounds':'ONC',
        'Organic oxygen compounds':'OOC',
        'Organoheterocyclic compounds':'OCL',
        'Organosulfur compounds':'OSC',
        'Others':'OTH',
        'Phenylpropanoids and polyketides':'PHP'}

print(t)

sns.set_style('ticks')
plt.figure(figsize=(10,10))
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm).plot()
ax = plt.subplot()
#annot=True to annotate cells, ftm='g' to disable scientific notation
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="crest", annot_kws={"size": 16});  
# use matplotlib.colorbar.Colorbar object
cbar = ax.collections[0].colorbar
# here set the labelsize by 16
cbar.ax.tick_params(labelsize=16)
ax.set_xlabel('Predicted labels',fontsize=20);ax.set_ylabel('True labels',fontsize=20); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(list([t_dict[i] for i in t]),fontsize = 16); ax.yaxis.set_ticklabels(list([t_dict[i] for i in t]),fontsize = 16)
plt.savefig('../output/mets_confusion_matrix_test.png',dpi=600)


ccs_mets_df = pd.read_csv('../output/ccs_mets_df.csv')
ccs_mets_df = ccs_mets_df[~ccs_mets_df['md'].isna()]
ccs_mets_df['md'] = ccs_mets_df['md'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
ccs_X = np.asarray([np.asarray(i) for i in ccs_mets_df['md'].values])
ccs_y = rf.predict(ccs_X)
ccs_y_label = le.inverse_transform(ccs_y)
ccs_mets_df['pred_label'] = ccs_y_label
ccs_mets_df['pred'] = ccs_y
ccs_mets_df.to_csv('../output/ccs_mets_df_class.csv',index=False)

embedding = umap.UMAP(random_state=42).fit_transform(ccs_X,y=ccs_y)
umap_df = pd.DataFrame(columns=['UMAP-1','UMAP-2','label'])
umap_df['UMAP-1'] = embedding[:,0]
umap_df['UMAP-2'] = embedding[:,1]
umap_df['Class'] = [t_dict[i] for i in ccs_y_label]
plt.figure(figsize=(10,10))
sns.scatterplot(data=umap_df, x="UMAP-1", y="UMAP-2", hue="Class",palette="deep")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('../output/mets_umap.png',dpi=600)

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(ccs_X,y=ccs_y)
tsne_df = pd.DataFrame(columns=['TSNE-1','TSNE-2','Class'])
tsne_df['TSNE-1'] = X_embedded[:,0]
tsne_df['TSNE-2'] = X_embedded[:,1]
tsne_df['Class'] = [t_dict[i] for i in ccs_y_label]

plt.figure(figsize=(10,10))
sns.scatterplot(data=tsne_df, x="TSNE-1", y="TSNE-2", hue="Class",palette="deep")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('../output/mets_tsne.png',dpi=600)
'''