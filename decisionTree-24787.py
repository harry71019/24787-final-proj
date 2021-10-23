import sys
import csv
import os
import numpy as np
import logging

# logging level (for debug)
logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

# input arguments
train_in = sys.argv[1]
test_in = sys.argv[2]
max_depth = int(sys.argv[3])
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]

# class instantiation for node
class Node:
    def __init__(self, key, depth, subDataset):
        self.left = None
        self.leftEdge = None
        self.right = None
        self.rightEdge = None
        self.key = key
        # depth = number of splits (first split = depth 1)
        self.depth = depth
        self.subDataset = subDataset
        # majority vote result for leaf node
        self.result = None
        self.result0Num = None
        self.result1Num = None

# class instantiation for attributes
class Attribute:
    def __init__(self, name, giniImpurity):
        self.name = name
        self.gini = giniImpurity

# helper functions
# Gini Impurity (binary)
def gini_impurity(num_result0, num_result1):
    if((num_result0+num_result1) == 0):
        return 0
    return 2 * (num_result0*num_result1) / (num_result0+num_result1)**2

def gini_impurity_dataset(dataset):
    dataset = np.array(dataset)
    results = get_attribute_cases(dataset[:,-1])
    num_result0 = 0
    num_result1 = 0
    for row in dataset:
        if(row[-1] == results[0]):
            num_result0 += 1
        elif(row[-1] == results[1]):
            num_result1 += 1
    if((num_result0+num_result1) == 0):
        return 0
    return 2 * (num_result0*num_result1) / (num_result0+num_result1)**2  

# get possible cases for selected attribute
def get_attribute_cases(attribute):
    cases = []
    for row in attribute:
        if(cases.count(row) == 0):
            cases.append(row)
        if(len(cases) == 2):
            if(cases[0]<cases[1]):
                temp = cases[0]
                cases[0] = cases[1]
                cases[1] = temp
            break
    return cases

# split dataset base on attribute (binary)
# returns two subsets
def split_dataset(dataset, attribute_index):
    subset0 = []
    subset1 = []
    # get possible cases for the attribute
    dataset = np.array(dataset)
    cases = get_attribute_cases(dataset[:,attribute_index])
    for row in dataset:
        if(row[attribute_index] == cases[0]):
            subset0.append(row)
        elif(row[attribute_index] == cases[1]):
            subset1.append(row)
    if(len(cases) == 1):
        cases.append(cases[0])
    return subset0, subset1, cases[0], cases[1]

# return attribute that has lowest gini impurity when choose to split
# input is current dataset and available attributes
# returns attribute with lowest GI and its GI
def lowest_gini_impurity_attribute(dataset, attributes, original_attributes):
    # get possible results
    dataset = np.array(dataset)
    results = get_attribute_cases(dataset[:,-1])
    # loop through all possible attributes
    each_gini_impurity = []
    for i in range (len(attributes)):
        # get possible cases for the attribute
        dataset = np.array(dataset)
        index = np.where(original_attributes == attributes[i])
        cases = get_attribute_cases(dataset[:,index])
        # number of Y/N after split
        num_result00 = 0
        num_result01 = 0
        num_result10 = 0
        num_result11 = 0
        # split base on cases
        for row in dataset:
            if(row[index] == cases[0]):
                if(row[-1] == results[0]):
                    num_result00 += 1
                elif(row[-1] == results[1]):
                    num_result01 += 1
            elif(row[index] == cases[1]):
                if(row[-1] == results[0]):
                    num_result10 += 1
                elif(row[-1] == results[1]):
                    num_result11 += 1
        # calculate TOTAL gini impurity
        weight0 = (num_result00+num_result01) / (num_result00+num_result01+num_result10+num_result11)
        weight1 = (num_result10+num_result11) / (num_result00+num_result01+num_result10+num_result11)
        total_gini_impurity = weight0*gini_impurity(num_result00,num_result01) + weight1*gini_impurity(num_result10,num_result11)
        each_gini_impurity.append(total_gini_impurity)
    # pick attribute with lowest gini impurity and specify the gini impurity
    min_attribute = attributes[np.where(np.isclose(each_gini_impurity,min(each_gini_impurity)))] #each_gini_impurity.index(min(each_gini_impurity))
    if(len(min_attribute)>1):
        min_attribute = min_attribute[0]
    min_gini = min(each_gini_impurity)
    return min_attribute, min_gini

# read dataset and load into numpy array
dataSet = np.genfromtxt(train_in, delimiter='\t', dtype=None, encoding=None)
title = dataSet[0,]
attributes = dataSet[0,:-1]
dataSet = dataSet[1:,]
unused_attributes = attributes[:]
original_attributes = attributes[:]
later_dataSet = dataSet[:] #backup for later use to put train data to tree
logging.debug(title)
logging.debug(dataSet)
logging.debug(attributes)
logging.debug(get_attribute_cases(dataSet[:,0]))
logging.debug("\n")

# function to grow decision tree
# based on current dataset, what attributes unused, max depth, grow from what node
def grow_decision_tree(dataSet, attributes, max_depth, parent_node, originalAtt):
    # check if depth exceeds max depth specified
    if(parent_node.depth == max_depth):
        logging.debug("max depth reached")
        logging.debug(parent_node.depth)
        return
    # check if there are still unused attributes
    elif(len(attributes) == 0):
        logging.debug("no more attributes")
        return
    # can continue growing
    else:
        min_attribute, min_gini = lowest_gini_impurity_attribute(dataSet,attributes,originalAtt)
        current_gini_impurity = gini_impurity_dataset(dataSet)
        if(current_gini_impurity == 0):
            logging.debug("data pure")
            return
        # gini gain > 0
        elif(current_gini_impurity - min_gini > 0):
            # split base on min attribute
            subset0, subset1, parent_node.leftEdge, parent_node.rightEdge = split_dataset(dataSet, np.where(originalAtt == min_attribute))#attributes.index(min_attribute)
            logging.debug(subset0)
            logging.debug(subset1)
            # assign attribute to parent node(indicating split on what attribute)
            parent_node.key = min_attribute
            logging.debug(min_attribute)
            # create child node
            parent_node.left = Node(None,parent_node.depth+1,subset0)
            parent_node.right = Node(None,parent_node.depth+1,subset1)
            # recursive grow tree on child
            grow_decision_tree(subset0, np.delete(attributes, np.where(attributes == min_attribute)), max_depth, parent_node.left,originalAtt)
            grow_decision_tree(subset1, np.delete(attributes, np.where(attributes == min_attribute)), max_depth, parent_node.right,originalAtt)
        return

# recursive print tree DFS (pre order)
def print_tree(root_node):
    if(root_node):
        print(root_node.key)
        print(root_node.subDataset)
        print(root_node.result)
        print(root_node.result0Num)
        print(root_node.result1Num)
        # recursive print left
        print_tree(root_node.left)
        # recursive print right
        print_tree(root_node.right)

# majority vote on subdataset
def majority_vote(subdataset,results_list):
    num_result0 = 0
    num_result1 = 0
    if(len(subdataset)==0):
        return None, 0, 0
    subdataset = np.array(subdataset)
    #results_list = get_attribute_cases(subdataset[:,-1])
    # for output printing order, does not effect function of decision tree
    if(len(results_list) == 2):
        if(results_list[0]>results_list[1]):
            temp = results_list[0]
            results_list[0] = results_list[1]
            results_list[1] = temp
    #####################################################
    for row in subdataset:
        if(row[-1] == results_list[0]):
            num_result0 += 1
        elif(row[-1] == results_list[1]):
            num_result1 += 1
    if(num_result0 > num_result1):
        return results_list[0], num_result0, num_result1
    # tie case (lex order, r > d)
    elif(num_result0 == num_result1):
        if(results_list[0]>results_list[1]):
            return results_list[0], num_result0, num_result1
        else:
            return results_list[1], num_result0, num_result1
    else:
        return results_list[1], num_result0, num_result1

# do majority vote on leaf nodes
def majority_leaf(root_node,results_list):
    if(root_node.key != None):
        root_node.result, root_node.result0Num, root_node.result1Num = majority_vote(root_node.subDataset,results_list)
        # recursive print left
        majority_leaf(root_node.left,results_list)
        # recursive print right
        majority_leaf(root_node.right,results_list)
    else:
        root_node.result, root_node.result0Num, root_node.result1Num = majority_vote(root_node.subDataset,results_list)

# read test data
test_dataSet = np.genfromtxt(test_in, delimiter='\t', dtype=None, encoding=None)
test_attributes = test_dataSet[0,:-1]
test_dataSet = test_dataSet[1:,]

# put test data into trained tree and create output
def test_into_tree(row, root_node, test_attributes,test_results):
    split_attribute = root_node.key
    index = np.where(test_attributes == split_attribute)
    if(row[index] == root_node.leftEdge):
        test_into_tree(row, root_node.left, test_attributes,test_results)
    elif(row[index] == root_node.rightEdge):
        test_into_tree(row, root_node.right, test_attributes,test_results)
    else:
        #print(root_node.result)
        test_results.append(root_node.result)

# calculate error metrics
def error(prediction, original):
    correct_num = 0
    wrong_num = 0
    for i in range (len(prediction)):
        if(prediction[i] == original[i]):
            correct_num += 1
        else:
            wrong_num += 1
    return wrong_num / (correct_num + wrong_num)

# print tree with homework format
def print_tree_lines(root_node,recur_depth,results_list):
    if(root_node.key != None):
        strRoot = "[{} {} / {} {}]".format(root_node.result0Num, results_list[0], root_node.result1Num, results_list[1])
        if(recur_depth == 1):
            print(strRoot)
        
        strLeft = "| "*recur_depth
        strLeft += "{} = {}: [{} {} / {} {}]".format(root_node.key[0],root_node.leftEdge[0],root_node.left.result0Num,results_list[0],root_node.left.result1Num,results_list[1])
        print(strLeft)
        # recursive print left
        print_tree_lines(root_node.left,recur_depth+1,results_list)
     
        # recursive print right
        strRight = "| "*recur_depth
        strRight += "{} = {}: [{} {} / {} {}]".format(root_node.key[0],root_node.rightEdge[0],root_node.right.result0Num,results_list[0],root_node.right.result1Num,results_list[1])
        print(strRight)
        print_tree_lines(root_node.right,recur_depth+1,results_list)
    else:
        pass



# grow tree if max depth > 0
# main function
if(max_depth>=0):
    root = Node(None, 0, dataSet)
    grow_decision_tree(root.subDataset, unused_attributes, max_depth, root, original_attributes)
    subdataset = np.array(root.subDataset)
    results_list = get_attribute_cases(subdataset[:,-1])
    if(len(results_list) == 2):
        if(results_list[0]<results_list[1]):
            temp = results_list[0]
            results_list[0] = results_list[1]
            results_list[1] = temp
    majority_leaf(root,results_list)
    
    # use train/test data on tree
    train_results = []
    test_results = []
    # get results using train data
    for row in later_dataSet:
        test_into_tree(row, root, test_attributes,train_results)
    # get results using test data
    for row in test_dataSet:
        test_into_tree(row, root, test_attributes,test_results)

# output train label
with open(train_out, 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    for row in train_results:
        writer.writerow([row])

# output test label
with open(test_out, 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    for row in test_results:
        writer.writerow([row])

# get output metrics
error_train = error(train_results, dataSet[:,-1])
#print(error_train)
error_test = error(test_results, test_dataSet[:,-1])
#print(error_test)

# output metrics textfile
with open(metrics_out, "w") as text_file:
    text_file.write("error(train): {0}".format(error_train))
    text_file.write("\n")
    text_file.write("error(test): {0}".format(error_test))

print_tree_lines(root,1,results_list)

'''
#test for tie majority vote
rList = ["notA","A"]
print(rList)
if(rList[0]>rList[1]):
    print("yes")
else:
    print("no")
'''



