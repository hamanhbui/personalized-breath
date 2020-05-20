import pickle
import numpy as np
from scipy import stats

breath_type="strong"
model_name="strong_multi_tcn_0"
list_test_acc=[]
list_test_eer_KNN=[]
list_test_eer_GMM=[]
    
with open('results/outputs/'+breath_type+'/list_test_acc_'+model_name, 'rb') as filehandle:
            list_test_acc = pickle.load(filehandle)
# with open('results/outputs/'+breath_type+'/list_test_eer_KNN_'+model_name, 'rb') as filehandle:
#             list_test_eer_KNN = pickle.load(filehandle)
# with open('results/outputs/'+breath_type+'/list_test_eer_GMM_'+model_name, 'rb') as filehandle:
#             list_test_eer_GMM = pickle.load(filehandle)

# breath_type="deep"
# model_name="deep_multi_tcn"
# list_test_acc_1=[]
    
# with open('results/outputs/'+breath_type+'/list_test_acc_'+model_name, 'rb') as filehandle:
#             list_test_acc_1 = pickle.load(filehandle)

# print(stats.ttest_ind(list_test_acc,list_test_acc_1, equal_var = False))
# print(len(list_test_eer_KNN))
# list_test_acc=list_test_acc[1:]
print(len(list_test_acc))
print(np.median(list_test_acc))
print(np.mean(list_test_acc))
print(np.std(list_test_acc))