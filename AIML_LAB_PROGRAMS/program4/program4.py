import pandas as pd
df_tennis=pd.read_csv('4.csv')
print("\n Given play Tennis Data Set:\n\n",df_tennis)

def entropy(probs):
    import math
    return sum([-prob*math.log(prob,2) for prob in probs])

def entropy_of_list(a_list):
    from collections import Counter
    cnt = Counter(x for x in a_list)
    num_instances=len(a_list)*1.0
    probs=[x/num_instances for x in cnt.values()]
    return entropy(probs)

def information_gain(df,split_attribute_name,target_atributes_name,trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs=len(df.index)*1.0
    df_agg_ent=df_split.agg({target_atributes_name:[entropy_of_list,lambda x:len(x)/nobs]})[target_atributes_name]
    df_agg_ent.columns=['Entropy','Propobservations']
    new_entropy = sum(df_agg_ent['Entropy']*df_agg_ent['Propobservations'])
    old_entropy = entropy_of_list(df[target_atributes_name])
    return old_entropy-new_entropy

def id3(df,target_atribute_name,atribute_names,default_class=None):
    from collections import Counter
    cnt = Counter(x for x in df[target_atribute_name])
    print(cnt)
    if len(cnt) ==1:
        print(len(cnt))
        return next(iter(cnt))
    elif df.empty or (not atribute_names):
        return default_class
    else:
        default_class=max(cnt.keys())
        gainz=[information_gain(df,attr,target_atribute_name) for attr in atribute_names]
        print("Gain =",gainz)
        index_of_max = gainz.index(max(gainz))
        best_attr=atribute_names[index_of_max]
        print("Best Attribute ",best_attr)
        tree = {best_attr:{}}
        remaining_attribute_names=[i for i in atribute_names if i!=best_attr]
        for  attr_val ,data_subset in df.groupby(best_attr):
            subtree=id3(data_subset,target_atribute_name,remaining_attribute_names,default_class)
            tree[best_attr][attr_val]=subtree
        return tree

atribute_names=list(df_tennis.columns)
print("List of atributes:",atribute_names)
atribute_names.remove('PlayTennis')
print("Predicting Attributes :",atribute_names)

from pprint import pprint
tree=id3(df_tennis,'PlayTennis',atribute_names)
print("\n\n The Resultant Decision Tree is : \n")
pprint(tree)

def classify(instance,tree,default = None ):
    attribute=next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result =tree[attribute][instance[attribute]]
        if isinstance(result,dict):
            return classify(instance,result)
        else:
            return result
    else:
        return default

test_data = pd.read_csv('4_test.csv')
test_data['predicted2']=test_data.apply(classify,axis=1,args=(tree,''))
print(test_data[['predicted2']])