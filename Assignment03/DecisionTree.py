from sklearn import tree
import pandas as pd
import graphviz
from sklearn.metrics import precision_score,f1_score, recall_score, accuracy_score

print("\n----------Task 4.1---------------\n")
# Manually typing up data and encoding variables
# home represents if a game was home or away. 1 indicates home
# top indicates if opponent in AP top 25 at Preseason. 1 indicates In
# media corresponds to media coverage. 1-NBC, 2-ESPN, 3-Fox, 4-ABC
train1 = {'home': [1, 1, 0, 0, 1, 0], 'top': [0, 1, 0, 0, 0, 0], 'media': [1, 1, 2, 3, 1, 4], 'result': [1, 0, 1, 1, 1, 1]}
test1 = {'home': [1, 1, 1, 0, 1, 0], 'top': [1, 0, 0, 1, 0, 1], 'media': [1, 1, 1, 4, 1, 4]}
trainDF1 = pd.DataFrame(data=train1)
testDF1 = pd.DataFrame(data=test1)

print("Entropy")
dt1 = tree.DecisionTreeClassifier(criterion="entropy")
trainedEntModel = dt1.fit(trainDF1.iloc[:, :3], trainDF1['result'])
tree.plot_tree(trainedEntModel)

train_dot_data1 = tree.export_graphviz(dt1, out_file=None, feature_names=list(trainDF1.columns.values)[:3],
                                       filled=True, rounded=True, special_characters=True)
entropyTrainGraph1 = graphviz.Source(train_dot_data1)
entropyTrainGraph1.render("EntropyTrainDecisionTree")
print("Training done.")

predictEntModel1 = trainedEntModel.predict(testDF1)
print("\tWin=1, Loss=0")
print("\tPrediction: " + str(predictEntModel1))
test_dot_data1 = tree.export_graphviz(dt1, out_file=None, feature_names=list(trainDF1.columns.values)[:3],
                                      filled=True, rounded=True, special_characters=True)
entropyTestGraph1 = graphviz.Source(test_dot_data1)
entropyTestGraph1.render("EntropyTestDecisionTree")
print("Testing done.")
print("Entropy finished.")

print("Gini")
dt1 = tree.DecisionTreeClassifier()
trainedGiniModel1 = dt1.fit(trainDF1.iloc[:, :3], trainDF1['result'])

tree.plot_tree(dt1.fit(trainDF1.iloc[:, :3], trainDF1['result']))

dot_data1 = tree.export_graphviz(dt1, out_file=None, feature_names=list(trainDF1.columns.values)[:3],
                                 filled=True, rounded=True, special_characters=True)
entropyGraph1 = graphviz.Source(dot_data1)
entropyGraph1.render("GiniTrainDecisionTree")
predictGiniModel1 = trainedGiniModel1.predict(testDF1)
print("\tWin=1, Loss=0")
print("\tPrediction: " + str(predictGiniModel1))
test_dot_data1 = tree.export_graphviz(dt1, out_file=None, feature_names=list(trainDF1.columns.values)[:3],
                                      filled=True, rounded=True, special_characters=True)
entropyTestGraph1 = graphviz.Source(test_dot_data1)
entropyTestGraph1.render("GiniTestDecisionTree")
print("Gini finished")

print("\n----------Task 4.2---------------\n")
# Manually typing up data and encoding variables
# weather data represents 1-Sunny, 2-Overcast, 3-rainy
# temp data represents 1-hot, 2-mild, 3-cool
# humidity data represents 1-High, 0-Normal
# windy data represents 1-True, 0-False
# result data represents if team should play. 1-yes, 0-no
train2 = {'weather': [1, 1, 2, 3, 3, 3, 2, 1, 1, 3, 1, 2, 2, 3],
          'temp': [1, 1, 1, 2, 3, 3, 3, 2, 3, 2, 2, 2, 1, 2],
          'humidity': [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
          'windy': [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
          'result': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]}
test2 = {'weather': [3], 'temp': [1], 'humidity': [1], 'windy': [0]}
trainDF2 = pd.DataFrame(data=train2)
testDF2 = pd.DataFrame(data=test2)
print("Entropy")
dt2 = tree.DecisionTreeClassifier(criterion="entropy")
trainedEntModel2 = dt2.fit(trainDF2.iloc[:, :4], trainDF2['result'])
tree.plot_tree(trainedEntModel2)

train_dot_data2 = tree.export_graphviz(dt2, out_file=None, feature_names=list(trainDF2.columns.values)[:4],
                                       filled=True, rounded=True, special_characters=True)
entropyTrainGraph2 = graphviz.Source(train_dot_data2)
entropyTrainGraph2.render("EntropyTrainWeatherDecisionTree")
print("Training done.")

predictEntModel2 = trainedEntModel.predict(testDF1)
print("\tWin=1, Loss=0")
print("\tPrediction: " + str(predictEntModel2))
test_dot_data2 = tree.export_graphviz(dt2, out_file=None, feature_names=list(trainDF2.columns.values)[:4],
                                      filled=True, rounded=True, special_characters=True)
entropyTestGraph2 = graphviz.Source(test_dot_data2)
entropyTestGraph2.render("EntropyTestWeatherDecisionTree")
print("Testing done.")
print("Entropy finished.")

print("Gini")
dt2 = tree.DecisionTreeClassifier()
trainedGiniModel2 = dt2.fit(trainDF2.iloc[:, :4], trainDF2['result'])

tree.plot_tree(dt2.fit(trainDF2.iloc[:, :4], trainDF2['result']))

dot_data2 = tree.export_graphviz(dt2, out_file=None, feature_names=list(trainDF2.columns.values)[:4],
                                 filled=True, rounded=True, special_characters=True)
entropyGraph2 = graphviz.Source(dot_data2)
entropyGraph2.render("GiniTrainWeatherDecisionTree")
predictGiniModel2 = trainedGiniModel2.predict(testDF2)
print("\tWin=1, Loss=0")
print("\tPrediction: " + str(predictGiniModel2))
test_dot_data2 = tree.export_graphviz(dt2, out_file=None, feature_names=list(trainDF2.columns.values)[:4],
                                      filled=True, rounded=True, special_characters=True)
entropyTestGraph = graphviz.Source(test_dot_data2)
entropyTestGraph.render("EntropyTestWeatherDecisionTree")
print("Gini finished")

print("\n---------Task 5.1---------------\n")
# Manually typing up data and encoding variables
# home represents if a game was home or away. 1 indicates home
# top indicates if opponent in AP top 25 at Preseason. 1 indicates In
# media corresponds to media coverage. 1-NBC, 2-ESPN, 3-Fox, 4-ABC
# win - 1, lose - 0
trainDF5 = {'home': [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
            'top': [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
            'media': [1, 4, 1, 1, 4, 1, 1, 4, 4, 1, 1, 3, 4, 1, 1, 1, 2, 4, 1, 1, 5, 1, 1, 4],
            'result': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, ]}
testDF5 = {'home': [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
           'top': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
           'media': [1, 1, 2, 3, 1, 4, 1, 1, 1, 4, 1, 4]}
y_true = [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
trainDF5 = pd.DataFrame(data=trainDF5)
testDF5 = pd.DataFrame(data=testDF5)
dt5 = tree.DecisionTreeClassifier(criterion="entropy")
entropyFitModel5 = dt5.fit(trainDF5.iloc[:, :3], trainDF5['result'])
tree.plot_tree(dt5.fit(trainDF5.iloc[:, :3], trainDF5['result']))
dot_data = tree.export_graphviz(dt5, out_file=None, feature_names=list(trainDF5.columns.values)[:3],
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Entropy_DecisionTree_Q5")
prediction = entropyFitModel5.predict(testDF5)

precision = precision_score(y_true, prediction)
F1_score = f1_score(y_true, prediction)
recall_score = recall_score(y_true, prediction)
accuracy_score = accuracy_score(y_true, prediction)
print("Precision Score: ", precision)
print("F1 Score ", F1_score)
print("Recall Score: ", recall_score)
print("Accuracy Score: ", accuracy_score)
print("Prediction for Entropy Q5:", prediction)

