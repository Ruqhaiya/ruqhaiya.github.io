---
title: "Analyzing and Predicting Dwelling Occupancy in Washington State"
excerpt: "The aim of this study is to predict whether a dwelling is occupied by owners or renters based on several features related to individual demographics and housing characteristics. We have used Support Vector Machines (SVM) to classify the dataset. Our models achieved up to 82% accuracy in classifying dwellings. It was observed that factors such as Age and Number of bedrooms were significant predictors in determining whether a dwelling is owned or rented followed by other predictors like average house income and cost of utilities. Among the models we built, the linear kernel was recommended for its robustness and simplicity. The findings of this study provides a thorough analysis of the factors influencing dwelling occupancy and uncover deeper patterns which will be useful to real estate professionals in understanding housing trends. "
collection: portfolio
---

# Analyzing and Predicting Dwelling Occupancy in Washington State

## Table of Contents
- [Analyzing and Predicting Dwelling Occupancy in Washington State](#analyzing-and-predicting-dwelling-occupancy-in-washington-state)
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
The aim of this study is to predict whether a dwelling is occupied by owners or renters based on several features related to individual demographics and housing characteristics. We have used Support Vector Machines (SVM) to classify the dataset. Our models achieved up to 82% accuracy in classifying dwellings. It was observed that factors such as Age and Number of bedrooms were significant predictors in determining whether a dwelling is owned or rented followed by other predictors like average house income and cost of utilities. Among the models we built, the linear kernel was recommended for its robustness and simplicity. The findings of this study provides a thorough analysis of the factors influencing dwelling occupancy and uncover deeper patterns which will be useful to real estate professionals in understanding housing trends. 

[Back to top](#table-of-contents)

---

## Introduction
Renters tend to skew toward the lower ends of the economic scale when it comes to income and wealth, according to data from the Federal Reserve’s 2019 Survey of Consumer Finances[1]. The primary goal of this study is to predict whether dwellings are occupied by owners or renters based on various demographic and housing-related factors. By using three different Support Vector Machines (SVM) kernels—linear, radial basis function (RBF), and polynomial, we not only seek to explore the predictive capabilities of these models but also gain insights into the underlying patterns within the data.

The dataset is obtained from the US Census, accessed through IPUMS USA[2]. This dataset is comprehensive with a wide range of variables including individual demographic information such as age, income, education level, and marital status as well as housing characteristics like electricity cost, year of construction, and population density of the surrounding area. These variables offer a rich source of information for understanding how individual attributes relate to whether people own or rent their homes. 

Throughout this report, we will explore the dataset and pre-process it. Pre-processing the data plays an essential role in generating insights that we can trust. The goal is to understand the application of SVMs to classification tasks and analyze how different kernel functions influence model performance. We will examine the relationships between selected variables and housing occupancy. 

[Back to top](#table-of-contents)

---

## Theoretical Background
Support Vector Machines(SVMS) are supervised learning methods used for classification and regression problems. SVM is a common term used to refer to the maximal margin classifier, support vector machine and the support vector machine. However, Support vector machine is a generalization of maximal margin classifier which requires data to be linearly separable. The support vector classifier is an extension of the maximal margin classifier which can be applied to a broader variety of datasets. The support vector machine is an extension of the support vector classifier and SVM can be used in cases where the data has a non-linear boundary. 

The support vector machine uses a hyperplane to separate the classes. A hyperplane in a p-dimensional space is a flat subspace of dimension p-1. In 2D, a hyperplane is a line and in 3D, it is a plane. When p>3, it’s hard to visualize a hyperplane but the concept of p-1 dimensional subspace still applies.  A hyperplane is in p-dimension is defined by the equation: 
β0 + β1X1 + β2X2 + · · · + βpXp = 0				eq(1)

A point X= (X1,X2, . . . ,Xp)T in p-dimensional space lies on the hyperplane if it satisfies the eq(1). If eq(1) is greater than 0, X is on one side of the hyperplane, and if it is less than 0, X is on the other side. However, if data is perfectly linearly separable, there can be infinite hyperplanes as shown in figure 1 [3]. 

![image](https://github.com/user-attachments/assets/4ba2984e-c497-4a6e-995b-be13a26e09db)

Figure 1
This is why choosing the hyperplane is crucial for the analysis, which is why we have the maximal margin classifier.  It is the optimal hyperplane– separating two classes–that is farthest from the training data. The SVM uses the principle of maximizing the distance between nearest data points from the hyperplane from either class. The distance between the hyperplane and the point is called as margin as shown 

The SVM operates on the idea of increasing the space between the closest points of each group and the dividing line, which is the hyperplane. The gap between this line and the nearest point is what we call the margin, as demonstrated in figure (2) [3].

![image](https://github.com/user-attachments/assets/11ec01b6-35bf-4d9a-b013-9768636fc298)

Figure 2

The nearest data points to the hyperplane are called the support vectors. Larger distance between the margin makes the model better. The mechanism of finding the hyperplane is fully accomplished by a subset of the training samples and the support vectors. Therefore, the support vectors play an essential role in determining the position of the hyperplane, and removing other training data points does not have any effect on the model but removing support vectors could affect our model performance drastically. They help in determining the optimal hyperplane for the dataset. 

A classifier based on separating hyperplanes is good until an observation arrives which is far from the hyperplane, this shifts the hyperplane  and results in a tiny margin. Then our confidence in the classification based on thin margin decreases. To tackle this, we have a Support vector classifier that is more robust to individual observations, although we make a trade-off here unlike in the Maximal Margin Classifier where we have a hard margin, here we allow few misclassifications and call it a soft-margin. We trade a small portion of our model’s performance to get better overall results. Rather than having the optimal maximal margin so that data are on the correct side of the hyperplane and margin, we instead allow some data to be on the incorrect side of the margin, or even the hyperplane as shown in Figure (3) [3].

 ![image](https://github.com/user-attachments/assets/cfaff2db-3bf8-47f0-aa0a-5192e031431b)

Figure 3: observations 11 and 12  are on the wrong side of the hyperplane and the wrong side of the margin.
This solution is defined as: ε_i ≥ 0, Σ_(i=1)^n ε_i ≤ C  			eq(2)

Where C is a tuning parameter. We try to find the highest possible C for our model, which means we want to allow misclassifications while also wanting our model to perform well. 
If ε_i = 0 then the ith observation is on the correct side of the margin, If ε_i  > 0 then the ith observation is on the wrong side of the margin, meaning it has violated the margin. If ε_i  > 1 then it is on the wrong side of the hyperplane itself. 

C tuning parameter is often referred to as the budget, because we have a certain budget to allow misclassifications. The value of c controls the tradeoff between the model's ability to fit the training data and its ability to generalize to new data. SVM's kernel hyperparameter is crucial for model performance, with options including linear, polynomial, and radial kernels. Different kernels enlarge the feature space in a specific way. Tuning these hyperparameters can have a significant effect on model accuracy.

Polynomial kernel is used when data is not linearly separable, so a polynomial function is used to separate the data. For instance, by increasing the feature space using quadratic, cubic, and even higher-order polynomial functions of the predictors.So instead of fitting a support vector classifier using p features X1,X2, . . . ,Xp, we instead fit a support vector classifier using 2p features X^1, X1^2, X^2, X2^2 , . . . ,Xp,Xp^2. The basic idea is to transform the input data into a higher-dimensional space where it can be linearly separated, and then build a linear model on top of it to separate the classes. The kernel function used in polynomial kernel SVM is defined as 
 [3]
where x and y are input feature vectors, c is a constant, and d is the degree of the polynomial. When d is 1, the polynomial kernel is just a linear kernel, and when d is higher, the kernel function puts the data into a higher-dimensional space. The degree parameter controls the complexity of the polynomial function used to separate the data. If the degree is too low, the model may not be able to separate the data effectively, and if the degree is too high, the model might be overfitting and perform poorly on new data. Tuning these hyperparameters is important for achieving good performance with polynomial kernel SVMs. 
The RBF kernel (Radial Basis Function), uses gamma to check if a new data point, let's say x^*, is near or far from the points we know. If x^* is far away, the radial kernel, with the help of gamma, basically says this point doesn't matter much for making predictions. This way, only the points that are really close to x^* influences what the prediction will be. RBF is defined as:
 
Why do we not enlarge space using the original features? Using kernels is computationally lighter, because SVMs are computationally expensive and enlarging features could create infinite dimensions. 
When there are more than 2 classes, the two most popular methods to use are the one-versus-one and one-versus-all.  One-versus-one creates a classifier for every possible pair of classes. If we have K classes, this means we will have K(K−1)/2 classifiers. Each classifier votes for an observation to belong to one of two classes. The class that gets the most votes across all classifiers is the final choice for where the observation fits.

The one-versus-all method, also known as one-versus-rest, is another approach where we create one classifier per class, comparing each class against all the others combined. Each classifier gives a confidence score for its class, and the one with the highest score decides the class for the observation.

[Back to top](#table-of-contents)

---

## Methodology
The dataset provided a unique challenge as each row represented a dwelling with multiple occupants living in the same house. SERIAL provides a unique identification number for each household in a dwelling and we observed that there are multiple records for a single household. More than 50% of the dataset is redundant with only ~30k unique households. 
We have subsetted the data by grouping each household by SERIAL column and only keeping the row which has the highest age, meaning the oldest individual of the household. This is because it is highly likely that they might be the renter or the owner.  

Then we analyzed the dataset further and decided to remove these columns for the reasons mentioned below:

●	PERWT: This variable is used when we want to do individual person analysis and it is not directly related to predicting home ownership, so having it is not required for our study.

●	BRTHYR: BRTHYR corresponds to the date of birth of a person and it was deleted as we have the AGE variable, this column is repetitive.

●	PERNUM: This column corresponds to the number of people in a family. It is directly not relevant since we are taking the eldest member from each family. But we will encode it like 1 for PERNUM > 1 and 0 for PERNUM ==1. This might be a good piece of information for our models. 

●	AGE: We took the maximum age from the rows to align it with our assumption that older individuals in the household will be the owner or renter. 

●	EDUCD : We deleted this column from the data set since it's correlated with the EDUC, it’s repetitive. 

●	INCTOT : We took the mean for this column as it gives us the average income of the household. 

●	HHINCOME : We removed this as it's highly correlated to average income INCTOT. 

After fitting the first model, it was noticed that the accuracy is 100% and it definitely means our model is overfitting. A decision tree classifier was used initially to analyze what’s causing this. We found that it was the VALUEH column that holds the value of each household, it was a perfect separator. It makes perfect sense that the prediction can be easily made with the help of this data. Hence, we removed this column from the dataset. 

Svm is not good with too many categories, so we converted the MARST to a binary variable, 0 if an individual is single and 1 otherwise. 
Then we addressed the data columns that included the cost of electricity, water, gas, and fuel. The predictors associated with these records had a particular code 9999, 9993, 9997 that indicated whether there was no cost or if these expenses were already included in the rent. It was crucial to replace these values with 0 to avoid model inaccuracy. After subsetting, although we had 30k rows but we used a sample of 10K 
rows because the SVM was taking a long time to compute, 10k is also a good size for a dataset.

We used SVM for classification tasks to accurately identify the houses, using a variety of demographic and environmental parameters as predictors such as age, education level, family income, and cost of maintaining a property. We used cross-validation techniques to assess the performance of our model. Later, we expanded the model to include the RBF kernel and polynomial kernel with varying cost, gamma, and polynomial degree values. 

We found that even after training the model with several hyperparameters, the results were comparable to what we were achieving with the linear SVM model although using the radial kernel, we achieved a slightly higher accuracy of 82%. Choosing the linear model as the final model is a good decision because it requires less computational effort to achieve almost similar outcomes. In summary, our process includes recognizing and addressing outliers, considering how to aggregate the data based on the unique features, and then training the SVM model, tuning it to choose the best parameters and evaluating its performance on the test dataset.

[Back to top](#table-of-contents)

---

## Computational Results
To build the model to classify the dwelling as rented or owned, we have used various variables like income, marital status, education and the cost which is involved in the household such as cost of gas, electricity and fuel. We have achieved 82% accuracy in classifying the dwelling. We have used linear, RBF and polynomial kernels to train the machine learning model. We cross validated the model using cost, gamma, and degree hyperparameters. It was observed that the model was giving almost similar accuracy with the RBF and linear kernel. Hence, it’s advised to use the linear kernel since it is a robust model that requires comparatively less computation power. A linear SVM model is also less prone to overfitting. 

We used a decision boundary plot to visualize the model’s performance. Plot for the linear SVM model is shown in figure 4. The plot was constructed using Age and Bedrooms features from the dataset that were significant for the model. Only two features were used because it's easier to depict how the SVM finds optimal boundaries between classes in a two-dimensional space.

 ![image](https://github.com/user-attachments/assets/da631f5b-1a5c-4eb7-8895-7f910179ef03)

Figure 4

The SVM model's decision boundary clearly shows a linear separation between homeowners and renters, achieving an accuracy of 81.2%. As age increases, depicted by higher values on the x-axis, the likelihood of owning a home also increases—this is visible as the blue region, representing homeowners, expands with age. Similarly, homes with more bedrooms are more likely to be owned, which is consistent with the expectation that larger families or more financially stable individuals tend to own their properties. These findings align with common observations that older individuals, who typically have greater financial stability, are more likely to own homes. The relationship between larger homes and homeownership also suggests that families needing more space are more inclined to buy rather than rent. 

 ![image](https://github.com/user-attachments/assets/7fae411e-6c27-4fad-a088-cc373feb122f)

Figure 5

The decision boundary surface for SVM with radial kernel is shown in figure 5. The non-linear boundary shows how the RBF kernel is able to capture complex patterns in the data. We can see how the orange region has made its way to the top left side of the plot, where age takes lower values and bedrooms take higher values. This conveys that younger adults who live in large dwellings are more likely to be classified as renters. Then as age and number of bedrooms increases, they’re more likely to be classified as an owner, the same as seen in linear SVM. 

Figure 6 shows the decision boundary for the SVM polynomial kernel. We can infer from the plot that it’s decent at capturing patterns in the data. As age and and house size increases to a ceratin value, the polynomial SVM model tends to classify it as a owner. 

 ![image](https://github.com/user-attachments/assets/756ef515-9ebe-4c3b-b8da-750823871085)

Figure 6
The models were working well on the test dataset. We got almost the same performance in terms of accuracy and error on the test set as we were getting on the training set. It conveys that the models were not overfitting although we can see that the RBF kernel boundary might be overfitting the data. 

Figure 6 shows the confusion matrix for the linear SVM, RBF and Polynomial SVM. Class 1 represents the owned category and class 2 represents the rented category. It is evident that the models were good at predicting when a dwelling is owned compared to class 2 but it is also a result of our dataset which is not perfectly balanced. We have more data for class 1 compared to class 2. The results are comparatively better since we have used stratified sampling to deal with the class imbalance. 
  
 ![image](https://github.com/user-attachments/assets/08eb168d-4004-4db0-b87d-fe06e1a3bf96) ![image](https://github.com/user-attachments/assets/a7d346d3-bb6d-46cf-8f6e-ce1229b4fb50) ![image](https://github.com/user-attachments/assets/6285b6d9-92e5-4d8b-8b98-079d91105ca9)

Figure 6 Confusion matrix for all three models: Linear, RBF, Polynomial SVM kernels

[Back to top](#table-of-contents)

---

## Discussion
It is worth noting that the IPUMS USA dataset includes a vast array of data that can be used to answer many more questions related to the dwelling’s ownership. This study concentrated specifically on the people's income, age, and education, which was just one component of the data. Regardless of this, our findings provide valuable insights into knowing which variables are important to determine if people will own the house or not. 

Apart from income and education, the study was based on several aspects such as the cost of maintaining the space such as electricity, fuel, and gas. Our findings show that predictors such as age, number of bedrooms, cost of utilities and average household income are strong predictors and have a considerable impact on a person's likelihood of homeownership. The data could be further studied by answering questions based on parental education status and income and determining whether or not the person will buy their own home later in life. We also want to see how the results differ if we only look at the data for married couples as a further extension of this report. 

Some of the insights or conclusions we can draw from the models are: 
1.	Younger adults seem to fall under the renter’s category, which means dwellings are not affordable for young adults who might have less financial stability.
2.	Older adults are more likely to be owners rather than renters. It was also observed that most of the dwellings that are bought are quite large which shows the preference of families. This could be helpful in real-estate planning. 
3.	It was also noticed that even though the houses are large but if they are occupied by young adults, they’re more likely to be classified as renters. Same goes for the older adults who live in smaller dwellings, they are more likely an owner regardless of the size of the dwelling. 
These insights show how our models were able to capture such deeper insights. On test results, the SVM model attained an accuracy of 81.2%, suggesting its usefulness in identifying the residence. Furthermore, adding the RBF or polynomial kernel had no significant effect on the model's accuracy with RBF just doing 82%, slightly higher. However, the study had limitations such as the cross-sectional nature of the data, which limits the ability to infer causality, and self-reported data because the data was too correlated, resulting in data loss of nearly 50%. We didn't have many distinct features to work with, such as bedrooms and rooms, which were significantly associated with each other. Other examples include the relationship between income and highest income, as well as education and education code. Furthermore, the sample size of the dataset was relatively small, which may limit the generalizability of findings to a larger population. 

[Back to top](#table-of-contents)

---

## Conclusion
This study focused on using Support Vector Machine (SVM) models and their kernel extensions, like RBF and polynomial, to predict ownership of a dwelling. We used data sourced from IPUMS USA, originally collected by the US Census. The results were promising, with the models reaching an accuracy of 82% (radial kernel) for the binary classification task of predicting home purchases versus rentals. This suggests that SVMs can be effective for real estate predictions. Features such as age, number of bedrooms, cost of utilities and average household income were strong predictors. The strong performance of the models could be beneficial for real estate agents and policymakers in developing strategies that serve both companies and individuals. For instance, these insights could be used in understanding and planning what type of homes should be prioritized such as family-sized based on number of bedrooms. It could help in potentially stabilizing the housing market and making homes more accessible. Additionally, this analysis could also be used in choosing the right location for new housing projects and increasing the chances of having more homeowners, based on a detailed analysis of the preferences individuals tend to have while buying a dwelling. This study significantly adds to the growing knowledge about applying SVMs with various kernels to homeownership classification tasks and analysis.

[Back to top](#table-of-contents)

---

## References
[1] Pew Research Center. (2022, March 23). Key facts about housing affordability in the U.S. Pew Research Center: Short Reads. https://www.pewresearch.org/short-reads/2022/03/23/key-facts-about-housing-affordability-in-the-u-s/

[2] Steven Ruggles, Sarah Flood, Matthew Sobek, Danika Brockman, Grace Cooper,  Stephanie Richards, and Megan Schouweiler. IPUMS USA: Version 13.0 [dataset]. Minneapolis, MN: IPUMS, 2023. https://doi.org/10.18128/D010.V13.0

[3] James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An Introduction to Statistical Learning with Applications in Python. (Original work published 2023) https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html

[Back to top](#table-of-contents)

---
