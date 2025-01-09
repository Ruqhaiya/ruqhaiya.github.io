---
title: "Analyzing-Youth-Substance-Use-Based-on-Family-Religious-and-Education-background"
excerpt: "This study leverages the National Survey on Drug Use and Health (NSDUH) to explore different aspects of youth drug use using decision trees and ensemble methods. This project addresses three key predictions: identifying whether an individual uses alcohol with the help of binary classification, then estimating the frequency of alcohol use over the past year using multi-class classification, and finally predicting the age at which an individual first consumed alcohol using regression techniques. From our analysis, several variables proved to be significant predictors across different models. Factors related to educational achievements (EDUSCHGRD2) and demographic factors such as race (NEWRACE2), income levels (INCOME), and lifetime marijuana use (YFLMJMO) and alcohol consumption standards (STNDALC) were influential in predicting youth drug use behaviors. The binary classification gradient boosting model achieved a good accuracy of 83.12%, however the multi-class classification and regression tasks showed relatively poor performance but nonetheless gave us some insights about the factors related to youth substance abuse. These findings highlight the crucial role of both socio-demographic variables and substance use history in predicting drug use. The findings advocate for the critical role of family background and education in mitigating risky behaviors and provide insights for improving preventive strategies."
collection: portfolio
---

- [Github Code Repository](https://github.com/Ruqhaiya/Analyzing-Youth-Substance-Use-Based-on-Family-Religious-and-Education-background)

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Methodology](#methodology)
- [Computational Results](#computational-results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

---  

## Abstract
This study leverages the National Survey on Drug Use and Health (NSDUH) to explore different aspects of youth drug use using decision trees and ensemble methods. This project addresses three key predictions: identifying whether an individual uses alcohol with the help of binary classification, then estimating the frequency of alcohol use over the past year using multi-class classification, and finally predicting the age at which an individual first consumed alcohol using regression techniques. From our analysis, several variables proved to be significant predictors across different models. Factors related to educational achievements (EDUSCHGRD2) and demographic factors such as race (NEWRACE2), income levels (INCOME), and lifetime marijuana use (YFLMJMO) and alcohol consumption standards (STNDALC) were influential in predicting youth drug use behaviors. The binary classification gradient boosting model achieved a good accuracy of 83.12%, however the multi-class classification and regression tasks showed relatively poor performance but nonetheless gave us some insights about the factors related to youth substance abuse. These findings highlight the crucial role of both socio-demographic variables and substance use history in predicting drug use. The findings advocate for the critical role of family background and education in mitigating risky behaviors and provide insights for improving preventive strategies.

[Back to top](#table-of-contents)

---

## Introduction
This study employs data from the National Survey on Drug Use and Health (NSDUH), an annual survey conducted to gather information about substance use and related behaviors from U.S. residents aged 12 and older. The dataset includes responses related to the use and frequency of the use of alcohol, marijuana, and tobacco. There is a group of variables that are responses to questions about youth’s attitude regarding substance use, what they feel about substance use, as well as how they feel about their peers involved in substance use, and how their parents feel about substance use, whether they have been praised by their teachers or parents. Along with this, the dataset also has responses to whether youth was involved in a fight, or theft as well as their involvement in religious events and participation in team building or leadership activities. 

Through this study, we aim to find important factors that can help us understand youth’s behavior toward alcohol. The outcomes of this research will provide valuable guidance for public health initiatives that are aimed at reducing substance use among youth. 

We have used decision trees and ensemble models to make three key predictions: 
1. Identifying Alcohol Use - Binary Classification: The first task is to predict whether an individual uses alcohol. This is a binary classification where the model will predict a 'yes' or 'no' based on various predictors related to demographic details, youth experiences etc. 
2. Estimating Frequency of Alcohol Use - Multi-Class Classification: The second task is where we are estimating the frequency of alcohol use over the past year. The model puts individuals into multiple groups based on how often they consumed alcohol in the last year.
3. Predicting Age of First Alcohol Use - Regression Analysis: The third task is to predict the age at which an individual consumed alcohol for the first time. This is done using regression techniques.

[Back to top](#table-of-contents)

---

## Theoretical Background
Decision tree is a supervised learning approach used for classification or regression tasks. Decision trees are built using a top-down greedy or recursive binary splitting. Initially all the observations belong to a single node at the top of the tree or the ‘root’ if we look at it upside down. Then it successively splits the predictor space, each split results in two new branches further down on the tree. It is called ‘greedy’ because at each step of the tree-building process, the best split is made at that particular step, rather than looking further ahead and selectively picking a split that will result in a better tree. The splits are made based on some splitting criterion such as Gini index– a measure of purity of the resulting split, For regression trees the splitting criterion is essentially a measure of residual squared sum, or RSS. The process of splitting goes on until a stopping criteria is reached, for instance, we may continue until no region contains more than five observations[2] or when no more improvement is expected in the performance. The stopping criterion can be specified in terms of tree depth and the optimal tree size is usually found using cross-validation with various tree sizes. 

Tuning for decision trees is essentially performing a cross validation and pruning it with the optimal tree size found via cross validation. Pruning a tree mainly reduces the complexity of the model which means there may not be any significant improvement in the accuracy of the model but the overall tree would be easier to comprehend. Decision tree models tend to perform poorly when the feature space is large but due their ease of interpretability, these models are quite robust. Datasets with many categorical variables are also suitable for decision tree models as we don’t necessarily need to encode them. To improve the predictive power, ensemble methods like bagging, random forest, gradient boosting come into the picture.

Bootstrap aggregation or bagging is an ensemble technique that is used to reduce the variance of a statistical learning method. It is used in situations where it is challenging to directly compute the standard deviation of a quantity. The decision trees suffer from high variance. This means that if we randomly split the training data into two different sets, and build a decision tree to both sets, the results obtained could be quite different. If we have a set of n independent observations Z1, . . . , Zn with variance  2, the variance of the mean Z of the observations is given by    2n. This simply means averaging a set of observations reduces variance. Hence, to reduce the variance and increase the test accuracy of a statistical learning method is to take many training sets from the dataset and build a separate model using each set and average the resulting predictions. Bagging is applied to regression trees by simply constructing B regression trees using B bootstrapped training sets, and averaging the resulting predictions. These trees are grown deep, and are not pruned. Hence each individual tree has high variance, but low bias. Averaging these B trees reduces the variance. Bagging has been demonstrated to give impressive improvements in accuracy by combining together hundreds or even thousands of trees into a single procedure[2]. In the context of classification problems, a simple bagging approach is to go by majority. For instance, for a given test observation, the class predicted by each of the B trees is recorded and a majority vote is taken, the final prediction is the most commonly occurring majority class among the B predictions. Although bagging is an improvement to decision trees but not when it comes to interpretability, hence we have something to the rescue called the ‘feature importance’. We can display a summary of the importance of each predictor variable using the RSS (for bagging regression trees) or the Gini index (for bagging classification trees). In the case of bagging regression trees, we can record the total amount that the RSS is decreased due to splits over a given predictor, averaged over all B trees. A large value indicates an important predictor. Similarly, in the context of bagging classification trees, we can add up the total amount that the Gini index is decreased by splits over a given predictor, averaged over all B trees.

Random forests provide a further improvement over bagged trees with a small tweak that decorrelates the trees. Similar to bagging, we build a number of decision trees on bootstrapped training sets. But when building these decision trees, at each split, a random sample of m predictors is chosen from the full set of p predictors. The split is allowed to use only one of those m predictors. A fresh sample of m predictors is taken at each split, and typically we choose mp meaning the number of predictors that are considered at each split are approximately equal to the square root of the total number of predictors. There’s an interesting rationale behind this, if there is a strong predictor, that predictor will most likely be in all the bagged trees, meaning all trees end up becoming similar to each other. By limiting the number of predictors, other less strong predictors will have a higher chance of being able to determine an outcome. The resulting average trees are more reliable because of this phenomenon. 

The ensemble models like bagging and random forest will produce an out-of-bag, OOB, error estimation, a measure of prediction error. When training these models, bootstrap samples of the original data are used, so the samples that are not used are called the “out-of-bag” samples. They are used to calculate the OOB error. Tuning this model will involve cross validating a range of m predictors to find the maximum number of features at each split.

In gradient boosting technique, the trees are grown successively, using a “slow” learning approach where information from the previous tree is used to improve the predictability of the decision tree. The current tree is trained in such a way that it corrects errors of the previous trees using residuals. There are three tuning parameters in the boosting method such as the learning rate, the number of trees and the number of splits in each tree that can be used.

[Back to top](#table-of-contents)

---

## Methodology

**Data Preparation**:

This study focuses on analyzing youth’s behavior based on factors related to their family, demographic details such as where they reside–urban or rural areas– and their experiences at school and religious beliefs. We removed data where the information about parents was either not known, deliberately not answered or the youth was 18+ years. We have also filtered out the data where the education background of an individual was unknown. Then data was further filtered to include individuals that belonged to densely populated or urban neighborhoods to address a particular group of youth that reside in urban areas. Features related to youth experiences’ such as involvement in religious events, praise from teachers had a few missing values across the dataset, hence rows where response was either unknown or not answered were excluded from the dataset as it might contribute to the noise in the dataset. The dataset was split using 80% of the original data for the training set and 20% for the testing set for all the classification and regression tasks.

**Models**:

We merged demographic indicators and features related to youth experiences and alcohol flag for the binary classification task. A Decision tree was used to predict whether the individual had ever used alcohol. To improve the model, pruning was implemented by finding the optimal tree size using cross-validation technique. 

Ensemble methods like random forest classifier, bagging and gradient booting were also used for the binary classification task. For all three methods, best parameters were found using grid search cross validation technique. For instance, an optimal number of maximum features was found for random forest, best n_estimators parameter and number of trees at each split for bagging was calculated and lastly, the optimal learning rate was found for the boosting method. Feature importances were analyzed for all the models. 

For multi-class classification tasks, we are estimating the frequency of alcohol use over the past year using the ‘ALCDAYS’ as the target variable which has 5 categories based on the number of days. Similar to the binary classification task, we used a decision tree classifier, pruned decision tree, random forest and boosting. All models were cross-validated to find the optimal parameters. 

For the regression task, the goal was to predict the age of initial alcohol use. The data was filtered to only include individuals over the age of 7 years and exclude where the response for the target feature IRALCAGE was unknown. Decision tree regressor, and ensemble methods like random forest and boosting regressor were employed to predict the age of initial alcohol consumption. Mean squared error was calculated for all the models since it is a regression task. Feature importances were analyzed to understand which predictors has the most influence. 

[Back to top](#table-of-contents)

---

## Computational Results

**Binary classification task**:

Table1 shows the results for the binary classification task which predicted whether an individual will use alcohol or not. The gradient boosting technique proved to be the best method for this task with an accuracy of 83.12% followed by other ensemble methods. The decision tree model performed decently after pruning it. 

| Model                          | Accuracy |
|--------------------------------|----------|
| Decision tree classifier       | 72.24%   |
| Pruned decision tree           | 79.18%   |
| Random forest classifier       | 81.39%   |
| Bagging                        | 80.91%   |
| Gradient boosting classifier   | 83.12%   |

Table1: Binary classification results

Table2

Figure1 shows the initial decision tree model which shows signs of overfitting since the size of the tree is huge. The confusion matrix for the decision tree in figure2 shows that the model is good at predicting outcome 0 compared to class 1. When cross validation was performed on the decision tree model, we can see a stable increase in the accuracy, which is a good indicator that the model is consistently performing well as shown in figure3. Figure4 and 5 represents the pruned decision tree and its confusion matrix, where a slight improvement was observed. The most important predictors for decision trees can be seen in table2 along with their importance. 

![image](https://github.com/user-attachments/assets/177b46a3-e6a2-4836-85b6-3af49eba5035)

Figure1 : Decision tree

![image](https://github.com/user-attachments/assets/a8ac2d7c-ab20-47d3-9f1b-523e74110974)

Figure 2: Confusion matrix - decision tree model

![image](https://github.com/user-attachments/assets/2dd22871-c639-465d-8eef-acf1fc2c01fd)

Figure3: Cross-validation results to find optimal tree size

![image](https://github.com/user-attachments/assets/5a0cc663-75a1-41f0-8899-522871f4f1de)

Figure4: Pruned decision tree

![image](https://github.com/user-attachments/assets/d5490b60-2f86-4e59-9802-5e90d154a1c8)

Figure5: Confusion matrix of pruned decision tree model
Random forest and bagging displayed good performance in terms of accuracy but they failed to predict class 1 and showed relatively worse performance for class 1 than decision trees. Although gradient boosting did exceptionally well in predicting both class 0 and 1. This shows that it was able to learn patterns in data better than the rest of the models. This comparison is displayed in figure 6,7 and 8. 

![image](https://github.com/user-attachments/assets/7ec2511e-f330-49b4-ad92-01cfdf6dda69)
 ![image](https://github.com/user-attachments/assets/a97fad8a-620e-42e6-8e3f-30fff1371b66)

Figure6: Confusion matrix of random forest method, Figure7: Confusion matrix of bagging method

![image](https://github.com/user-attachments/assets/ea9482bc-5730-4f44-a2d9-1faa6b64dbad)

Figure8: Confusion matrix of gradient boosting model

Multi-class classification task:

**Multi-class classification results**:

| Model                                    | Accuracy |
|------------------------------------------|----------|
| Decision tree classifier                 | 68.45%   |
| Pruned decision tree                     | 79.50%   |
| Pruned decision tree with class weights  | 60.88%   |
| Random forest classifier                 | 79.50%   |
| Gradient boosting classifier             | 78.08%   |


Table3 Multi-class classification results
The results for the multi-class classification task are shown in Table3. Contrary to the binary classification task, here the pruned decision tree, random forest and boosting model proved to be better in terms of accuracy. Although when we look at the confusion matrices for each one of these in figure 10,12 ad 13, we can notice that these models were only good at predicting ‘never used alcohol in the past year’ outcome(class 0). They were extremely bad at capturing the patterns to predict other classes. But when class weights were used for pruned decision trees, we observed slightly better predictions for other classes like class 1 and 2. 

![image](https://github.com/user-attachments/assets/dc1ef3b8-582f-44b2-b77a-d34685052ca7)

Figure 9 CV results for multi-class

![image](https://github.com/user-attachments/assets/064a7b6a-1696-4108-af7a-57d523183479)

Figure 10 confusion matrix for initial decision tree

![image](https://github.com/user-attachments/assets/92498608-1367-4188-b943-31aac6870529)

Figure 11 pruned decision tree
The same predictors seem to be important for the multi-class classification using decision trees as we noticed in the binary classification task. However, ensemble methods had different predictors that proved to be significant in the prediction of frequency of alcohol used in the past year as shown in table4. 

![image](https://github.com/user-attachments/assets/80c3807d-92d2-49b5-9093-f996d3eb0a9b)

Table4 Feature importances for random forest model

![image](https://github.com/user-attachments/assets/adb03b80-806f-4b64-9ae6-6f6c59217e94)

Figure 12 confusion matrix for pruned decision tree with class weights

![image](https://github.com/user-attachments/assets/604859c1-a7f6-426c-8826-aeda1232fc12)

Figure 13 confusion matrix for random forest model

**Regression task**:

The regression task was performed where the age of initial alcohol consumption was predicted using decision tree, random forest and gradient boosting method. Mean squared error was calculated to find how well the model performed which is shown in table 5 below. The cross validation results are shown in figure 14 for the pruned decision tree. Random forest and decision tree model with optimal tree size showed better performance compared to the rest. The feature importance of random forest is shown in figure 16 with education feature at the top followed by other predictors. The plot in figure 15 shows the actual versus the predicted values by random forest model. The data points don’t cluster perfectly around the diagonal line. This suggests that the model is not making very good predictions.

**Regression results**       

| Model                          | MSE    |
|--------------------------------|--------|
| Decision tree classifier       | 5.350  |
| Pruned decision tree           | 2.525  |
| Random forest classifier       | 2.780  |
| Gradient boosting classifier   | 3.060  |

![image](https://github.com/user-attachments/assets/03ea13d5-be32-41de-90b8-fa535503b91e)

Figure 14 cv results - pruned decision tree

![image](https://github.com/user-attachments/assets/372ed2c2-468f-41ff-a9d4-c86d57501b43)

Figure 15 random forest model 

![image](https://github.com/user-attachments/assets/5b9a6dc5-ba78-4987-b939-cdca759b9814)

Figure 16 Feature importance - random forest model

[Back to top](#table-of-contents)

---

## Discussion
The NSDUH dataset presents a comprehensive overview of behavioral and demographic variables that offer valuable insights into substance use patterns. Our analysis revealed that youth’s feelings about substance use, exposure to other substances like marijuana, parental communication about substance use, the child’s academic achievements, and their attitudes towards school play pivotal roles in influencing their likelihood of consuming alcohol. Specifically, positive school experiences and open conversations about substance risks are linked to a lower likelihood of substance use, underscoring the importance of supportive educational and familial environments.

The gradient boosting model, which was cross-validated to optimize performance, achieved an accuracy rate of 83.12% in predicting whether an individual would consume alcohol. This indicates the model's effectiveness in capturing the complex patterns in factors that contribute to alcohol consumption among youth.

It is not clear why the multi-class or regression task did not performing well given the array of significant predictors, perhaps it might be due to the class imbalance that lead to the models not fully learn other outcomes like in multi class classification, the models performed really well in predicting the outcome ‘never used alcohol in the past year’ but failed to accurately predict other classes. The data representing the class 0(never used alcohol) was significantly more when compared to other classes. A dataset that’s equally representative of all classes might be a better one for the multi-class and regression task. Although the regression models did well in terms of their MSEs, the plot of actual vs predicted values clearly shows otherwise. 

It is also important to acknowledge certain limitations in our study, the reliance on self-reported data might introduce biases, which can affect the accuracy of our predictions. The relatively small sample size could also limit the generalizability of our findings to a broader population.

Despite these limitations, our study provides critical insights into the factors that can reduce the likelihood of alcohol and other substance use among youth. These insights reinforce the value of targeted educational and family-oriented interventions. Extending this research to include longitudinal data could help clarify causal relationships and enhance the predictive power of our models. Incorporating a larger and more diverse dataset could improve the generalizability of our results, providing a stronger basis for developing nationwide substance abuse prevention programs.

[Back to top](#table-of-contents)

---

## Conclusion
Our research utilized decision tree models and tree-based ensemble methods, including Random Forest, Bagging, and Boosting, to predict alcohol consumption among youth. Employing data from the NSDUH survey, our models achieved commendable accuracies for binary classification tasks, highlighting their effectiveness in substance use prediction.

In binary classification tasks, the gradient boosting model notably achieved an accuracy of 83.12%. The multi-class classification task was only good at predicting if an individual has never used alcohol in the past year, the highest accuracy of 79.50% was achieved by pruned decision tree and random forest model. The regression task on the other hand has showed decent performance with least MSE of 2.58 with pruned decision tree model followed by 2.78 for the random forest model. 

Our findings are significant as they affirm that both decision tree models and ensemble techniques are robust tools for predicting whether an individual might engage in alcohol consumption. Additional studies could expand on this work by incorporating more diverse datasets or by applying these techniques to predict other types of substance use and behavioral outcomes. 

Overall, the effectiveness of our predictive models provides valuable insights for public health officials and educators seeking to implement evidence-based strategies to reduce alcohol misuse among youth. As we continue to refine these models and enhance their accuracy, they hold the potential to significantly impact public health initiatives by enabling more precise and timely interventions.

[Back to top](#table-of-contents)

---

## References
1. Substance Abuse and Mental Health Services Administration. (2020). National Survey on Drug Use and Health (NSDUH) 2020. [Codebook]. Retrieved from https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH2020/NSDUH-2020-datasets/NSDUH-2020-DS0001/NSDUH-2020-DS0001-info/NSDUH2020-DS0001-info-codebook.pdf.

2. James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An Introduction to Statistical Learning with Applications in Python. (Original work published 2023) https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html


[Back to top](#table-of-contents)

---
