---
title: "Analyzing Civil Servants data using Unsupervised Learning techniques"
excerpt: "This report leverages unsupervised learning techniques, specifically Principal Component Analysis (PCA) and k-means clustering algorithms, to explore the annual declarations of personal interest and investment data from Mongolian civil servants[1]. The dataset, originally scraped from a dynamic web platform and collected for investigative journalism training by the Mongolian Data Club, contains comprehensive information from 2016 to 2021. For this analysis, data from the year 2021 was utilized. PCA was employed to reduce the dimensionality of the dataset, capturing approximately 81.64% of the total variance with the first 16 components. Subsequently, k-means clustering was applied to the principal components, identifying distinct clusters within the data. This approach aimed to uncover underlying trends and group similar declarations, a quantification task that has not previously been attempted on this dataset."
collection: portfolio
---

- [Github Code Repository](https://github.com/Ruqhaiya/Analyzing-Civil-Servants-data-using-Unsupervised-Learning-techniques)

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
This report leverages unsupervised learning techniques, specifically Principal Component Analysis (PCA) and k-means clustering algorithms, to explore the annual declarations of personal interest and investment data from Mongolian civil servants[1]. The dataset, originally scraped from a dynamic web platform and collected for investigative journalism training by the Mongolian Data Club, contains comprehensive information from 2016 to 2021. For this analysis, data from the year 2021 was utilized. PCA was employed to reduce the dimensionality of the dataset, capturing approximately 81.64% of the total variance with the first 16 components. Subsequently, k-means clustering was applied to the principal components, identifying distinct clusters within the data. This approach aimed to uncover underlying trends and group similar declarations, a quantification task that has not previously been attempted on this dataset.

[Back to top](#table-of-contents)

---

## Introduction
Mongolia, situated in East Asia, is a nation characterized by its rich natural resources and expansive land. The economy is predominantly natural resource-based, with coal and copper being the major exports. The country's reliance on these resources makes its economy particularly vulnerable to fluctuations in global commodity prices[2]. Additionally, the heavy dependence on natural resources has led to significant challenges in governance, with the public sector, especially government agencies, being prone to corruption.

To address these issues and foster greater transparency, a comprehensive reform has been initiated by the Mongolian government, creating an Independent Authority Against Corruption[3]. This reform requires civil servants, starting from senior levels and elected representatives, to annually declare their personal interests and investments[1]. The goal is to increase accountability and build public trust in governmental institutions. These declarations are intended to provide a clearer picture of potential conflicts of interest and ensure that officials are acting in the best interest of the public.

The dataset for this initiative was carefully prepared from a publicly available source, originally scraped from a dynamic web platform that publishes civil servants' personal interest and investment declarations. This data, initially collected for use in the Mongolian Data Club's[4] investigative journalism training, contains thirty-seven features per individual report. These features include the reporter's last name, full first name, income, family and household income, and detailed information about real estate holdings, such as the number of apartments, houses, or land owned. It also covers other assets like the number of cars and their values. A unique aspect of this declaration is that it is self-reported; however, if any false information is provided, the individual can be held legally accountable.

In this report, unsupervised learning techniques, specifically Principal Component Analysis (PCA) and the k-means clustering algorithm, are utilized to analyze the data collected from these declarations. PCA is used to reduce the dimensionality of the data, making it easier to visualize and interpret, while k-means clustering aids in identifying patterns and grouping similar declarations together. Through these analyses, underlying trends in the data are aimed to be uncovered. This quantifying job has never been attempted on this dataset.

[Back to top](#table-of-contents)

---

## Theoretical Background

Unsupervised learning involves various statistical techniques used when we only have a set of features, such as X1, X2,...,Xp, measured across n observations. Unlike in supervised learning, we don't have an outcome variable to predict. Instead, our goal is to explore and understand the data itself. We aim to find interesting patterns in the data, visualize the data, and identify any subgroups or clusters among the observations. With unsupervised learning techniques, we can uncover hidden structures and gain deeper insights from the data even without a specific predefined outcome. Two particular types of unsupervised learning are principal components analysis, a tool used for data visualization or data pre-processing before supervised techniques are applied, and clustering, a broad class of methods for discovering unknown subgroups in data. 

**Principal components analysis**
Principal components analysis (PCA) is used to compute principal components and utilize them to understand the data better. PCA deals with a set of features, X1, X2,...,Xp, with no associated response variable, Y.  Apart from producing derived variables for use in supervised learning problems, PCA also serves as a tool for data visualization (visualization of the observations or visualization of the variables). It is also used as a tool for data imputation — that is, for filling in missing values in a data matrix.

Suppose we have a dataset with n observations and a set of features (like X1, X2,...,Xp) and we want to visualize these observations to perform an exploratory data analysis on the entire dataset. We can do this by creating scatterplots for each pair of features. However, with a large number of features, this quickly becomes overwhelming and impractical. For example, if we have 10 features, we would need to create 45 different scatterplots! In general, the number of scatterplots required is given by p(p−1)/2, where p is the number of features.

Clearly, a better method is required to visualize the n observations when p is large, especially when we aim to retain as much information as possible. This is where PCA comes into play. PCA helps us find a lower-dimensional representation of the data that captures the most important variations. The goal of PCA is to summarize the data into a few key dimensions. These dimensions, called principal components, are new features created from the original ones. The first principal component captures the direction of the greatest variance in the data. In simple terms, it's the line along which the data varies the most.

To compute these principal components, PCA looks for a linear combination of the original features. The first principal component, Z1, is found by combining the features in such a way that it explains the maximum variance. The coefficients used in this combination are called loadings, and they are normalized to ensure the combination is meaningful. 

Z1 = ϕ11X1 + ϕ21X2 +⋯+ ϕp1Xp

Given a dataset, we can find the first principal component by ensuring the features have a mean of zero and then calculating the combination of features that capture the most variation. This process is repeated to find subsequent principal components, each capturing the next highest variance while being orthogonal (independent) to the previous ones. 

zi1=ϕ11xi1+ϕ21xi2+⋯+ϕp1xip (first PCA)

zi2=ϕ12xi2+ϕ22xi2+⋯+ϕp2xip (second PCA)

Once we have computed the principal components, we can plot them against each other in order to produce low-dimensional views of the data. For instance, we can plot the score vector Z1 against Z2, Z1 against Z3, Z2 against Z3, and so forth. Geometrically, this amounts to projecting the original data down onto the subspace spanned by φ1, φ2, and φ3, and plotting the projected points.

Proportion of Variance Explained (PVE) is a measure used in Principal Components Analysis (PCA) to quantify how much of the total variability in the data is captured by each principal component. The PVE by each principal component indicates the fraction of the total variance in the data that is explained by that component. This is calculated by dividing the variance explained by a principal component by the total variance. The sum of squared deviations from the mean (TSS) represents the total variance, and the residual sum of squares (RSS) represents the variance not explained by the selected principal components. 

The PVE can be computed as 1-RSS/TSS  , and it is analogous to the R-squared value in regression analysis. By examining the PVE, one can determine the number of principal components needed to adequately capture the structure of the data, thus simplifying the analysis while retaining most of the original information. 

**Singular Value Decomposition**

SVD stands for Singular Value Decomposition, which is a mathematical technique used for matrix factorization. It decomposes a matrix into three separate matrices, allowing valuable information about the matrix's structure and properties.

Given an m x n matrix A, the Singular Value Decomposition factorizes it into three matrices:

A = U Σ V^T

where:
- U is an m x m orthogonal matrix, whose columns are called the left singular vectors of A.
- Σ is an m x n diagonal matrix, with diagonal entries known as the singular values of A. The diagonal elements are arranged in descending order.
- V^T is the transpose of an n x n orthogonal matrix V, whose columns are called the right singular vectors of A.

PCA is equivalent to SVD when the data is pre-scaled in the SVD decomposition. Hence, SVD can be used to reduce the dimensionality of a dataset by selecting the top k singular values and their corresponding singular vectors. This technique is especially useful for large datasets and can help with noise reduction and extracting the most informative features. SVD can reveal the most common directions or qualities the data have.

**Clustering**

Clustering is used to find subgroups or clusters within a dataset. The primary goal of clustering is to partition the observations into distinct groups such that the observations within each group are similar to each other, while observations in different groups are quite different. The definition of "similar" and "different" can vary depending on the domain and the specific characteristics of the data. 

There are many clustering methods available, but two of the most well-known are K-means clustering and hierarchical clustering. K-means clustering partitions the observations into a predefined number of clusters, while hierarchical clustering does not require specifying the number of clusters in advance. Instead, it produces a tree-like visual representation called a dendrogram, which shows the clusterings for all possible numbers of clusters, from 1 to n. Each method has its advantages and disadvantages, and the choice between them depends on the specific requirements and characteristics of the dataset. 

Let's imagine we have a dataset with n observations, each having p features. What K-means does is try to minimize the differences within each cluster. It does this by solving an optimization problem that aims to reduce the total within-cluster variation, making sure the points in each cluster are as close to each other as possible.

The algorithm works iteratively, first it randomly assigns each observation to one of the K clusters. Then, it calculates the centroid of each cluster, which is like finding the average position of all the points in that cluster. Next, it reassigns each observation to the cluster with the nearest centroid. This process keeps repeating until the cluster assignments stop changing. At each step, the within-cluster variation decreases, meaning the clusters become more distinct and well-defined.

Visualizing the results of K-means clustering can help understand its effectiveness. For example, as shown in Figure 1, the results of applying K-means to a simulated dataset with 150 observations in two dimensions for different values of K can be seen. Each observation is colored based on its cluster assignment, showing how the algorithm divides the data into distinct groups.

![image](https://github.com/user-attachments/assets/b566fac2-5e1d-476b-b436-bfa7a877f0c7)

Figure 1
One of the challenges with K-means clustering is deciding the number of clusters. Also, since the algorithm finds a local optimum rather than a global optimum, the final clustering can depend on the initial random assignment of clusters. This is why it's important to run the algorithm multiple times with different starting points and choose the best result based on the lowest within-cluster variation.

![image](https://github.com/user-attachments/assets/0b8b98d0-5b6a-4124-941b-2e8a9a221ada)

Figure 2

Figures 2 and 3 illustrate the progression and variability of the K-means algorithm. Figure 2 shows how the cluster centroids and assignments evolve over iterations, and Figure3 shows the results of running K-means six times with different initial assignments, emphasizing the need to choose the best solution.

![image](https://github.com/user-attachments/assets/e91a8afe-1608-4811-b298-93a35866aa36)

Figure 3

Hierarchical clustering is a method used to group observations into clusters, and it has the advantage of not requiring us to pre-specify the number of clusters. This technique creates a tree-based representation called a dendrogram, which visually displays how observations are grouped at various levels of similarity.
The most common hierarchical clustering method is bottom-up or agglomerative clustering. This method starts by treating each observation as its own cluster. Then, it repeatedly merges the two most similar clusters until all the observations are merged into a single cluster. The resulting dendrogram, which looks like an upside-down tree, shows the hierarchy of clusters formed at each step.

 ![image](https://github.com/user-attachments/assets/3e35914a-fea0-4b3a-bf6a-d9b3b239a911)

Figure 4
For instance, Figure 4 shows a dataset with 45 observations plotted in two-dimensional space. Even though the true class labels are known, we ignore them and use hierarchical clustering to find natural groupings. The resulting dendrogram shows how the observations are grouped based on their similarities as seen in figure 5. To identify distinct clusters, we can make horizontal cuts across the dendrogram, cutting the dendrogram at a height of nine results in two clusters, while a cut at a height of five results in three clusters. This flexibility allows us to choose the number of clusters based on the structure of the dendrogram. 

 ![image](https://github.com/user-attachments/assets/9c34e0c5-f561-4eed-8af1-77b4666f8e8a)

Figure 5
The algorithm for hierarchical clustering is quite straightforward. It starts with each observation as its own cluster, it calculates the pairwise dissimilarities using Euclidean distance nd merges the two most similar clusters in each step. This process continues until all observations are combined into a single cluster. 

The method we choose to link clusters together plays a big role in how the dendrogram looks. Some common linkage methods are complete, average, single, and centroid linkage. Each of these methods has its own pros and cons, and the choice can affect how balanced and easy to interpret the dendrogram is.

[Back to top](#table-of-contents)

---

## Methodology
Data preprocessing

The original dataset was scraped directly from the source site using a Selenium web driver[4] in a headless configuration. To avoid triggering DDoS protection and affecting the regular operation of the host server, the scraping was done from multiple virtual machines with random sleeping intervals, mimicking ten active users searching the declaration web. All column names were named after equivalent English names. The numerical data was self-explanatory and made sense; the only thing to be discussed was the unit, for example, how many millions or billions in Mongolian currency. To enable use for the project, job titles were translated into English using OpenAI's GPT-4o model via API calls. The API [5] implementation followed the developer's guidebook in Python. Dozen test API calls were made to set up the translation workflow. For example, in the preparation, the most approved words were asked and corrected as needed. The query was made on every row, one by one, using Python's mapping function. Additionally, for future projects, a feature column will be created to note the category of the positions.

The dataset used in this analysis was carefully prepared from a publicly available source, originally scraped from a dynamic web platform that publishes civil servants' personal interests and investment declarations. This data was initially collected for use in the Mongolian Data Club's investigative journalism training. The original dataset contained declarations from 2016 to 2021, but for this analysis, only data from 2021 was used.

To begin, the position names in the dataset, originally in Mongolian, were translated into English using OpenAI's API. In addition to the translation, potential categories were labeled for the translated job positions for reference purposes, although these categories were not utilized in the subsequent analysis. Similarly, the names of the institutions were translated into English using both OpenAI and the ChatGPT-4.0 model.

A notable challenge arose with duplicates following the translations. Some unique Mongolian inputs resulted in similar English translations. To preserve the integrity of the original dataset and prevent the loss of potentially significant information, no columns were removed despite the presence of these duplicates. This approach ensured that the data retained the nuances and specificities of the original entries, staying true to the dataset's original form. The final step in the data preparation involved removing NaN values, scaling the columns, and imputing missing values by the mean.

**Principal Component Analysis and K-Means Clustering**

Principal Component Analysis was used to reduce the dimensionality of the dataset, and k-means clustering algorithms were used to identify any pattern from the dimension reduced dataset.

The initial step involved running PCA to determine how many components were necessary to explain the variance in the data. This was achieved by performing PCA on the scaled dataset and visualizing the explained variance. Subsequently, a PCA biplot was created to visualize the principal component loading vectors. This biplot illustrated how each original variable contributed to the first two principal components, aiding in understanding the relationships between the variables and the principal components.

Following PCA, K-Means Clustering was applied to the principal components to identify clusters within the data.  The final step involved performing K-Means clustering with the determined number of clusters. The clusters were then visualized in the first two principal components, providing a clear representation of how the different declarations were grouped based on their features.

[Back to top](#table-of-contents)

---

## Computational Results

![image](https://github.com/user-attachments/assets/898df90a-0db2-408d-a2cd-fa6f3f5b7b46)

Figure 6
The sharp initial drop followed by a gradual decline shows that the first few principal components capture most of the important variations in the data. This means we can reduce the data to fewer dimensions without losing much information. It means that by focusing on the first two to five components, we can simplify the dataset while keeping its main structure intact. We plot the data further using PCA to see how much variance is explained by the first two components and the number of components.

  ![image](https://github.com/user-attachments/assets/0e5b16f2-e89b-4e11-a61f-7fadfc271aa8) ![image](https://github.com/user-attachments/assets/d5e6fd85-bb13-4af3-817d-2f3e568c7b27)

Figure 7,8
Figure 6 shows how much of the data's variability is captured by each principal component. The first principal component explains the most variance around 12%, and each subsequent component explains progressively less. Figure 7 shows the cumulative amount of variance explained as 20 principal components are included. As we move to the right, we see how adding more components captures more of the total variance. It helps identify the point where adding more components does not significantly increase the explained variance.

 ![image](https://github.com/user-attachments/assets/79cbec53-57a5-470d-84e5-df2e1b121aa6)

Figure 8
The plot on the left shows how much variance each principal component explains individually and the plot on the right shows the cumulative variance explained as more principal components are included. 

Figure 9
The PCA biplot visualizes both the observations and the variables in the first two principal component spaces. The arrows represent the original variables and it shows their direction and contribution to the principal components. 
 ![image](https://github.com/user-attachments/assets/7fbaa95f-7c4a-416d-a2c0-c23624b3199f)

![image](https://github.com/user-attachments/assets/a50f6103-1d5a-40a9-aa23-4448ab0f8aad)

 Figure 10
This scatter plot shows the results of clustering the data into three clusters using the first two principal components from PCA. Each point represents an observation, colored by its assigned cluster. This helps us visualize how the data is grouped and the separation between clusters.

 ![image](https://github.com/user-attachments/assets/a3b31934-31c5-40ac-b632-fecf655933ee)

Figure 11

 ![image](https://github.com/user-attachments/assets/ca6c0c21-aee6-4ff4-85dd-e460c8bd215b)

Figure 12

[Back to top](#table-of-contents)

---

## Discussion

Scree Plot
![image](https://github.com/user-attachments/assets/6df54c99-546a-4a5c-8f49-372637f67e64)

 
The scree plot displayed a sharp decline after the first principal component and then a more gradual decline. This "elbow" indicates that the majority of the variance is captured by the first few components. Specifically, the elbow suggests that the first 3-4 principal components account for a significant portion of the total variance. What does it means about the decomposition? It suggests that the dimensionality of our data can be reduced significantly without a substantial loss of information. By retaining only the first few principal components, we can capture the most critical aspects of the data.

Interpretation of U and V* matrix from the SVD (or "x" and "rotation" from PCA)
The U matrix shows how the original data points are represented in terms of the principal components. For example, we have data on various building counts and values, the U matrix shows how these data points cluster in the new space defined by the principal components. The V* matrix provides insights into how each original feature (such as different types of building or vehicle count) contributes to the principal components. 
Plotting first two principal components vs plotting on two original features
From the PCA biplot in figure 9, we can see that COUNT_HOUSE_BUILDING and TOTAL_CONSTRUCTION_VALUE were significant contributors to the first two principal components. So we used these two features for our comparison. We plotted the data using these original features and first two principal components. 

  ![image](https://github.com/user-attachments/assets/0df6dabb-102c-4e06-b8df-07fbfffd9434) ![image](https://github.com/user-attachments/assets/c1f2da19-7213-4d47-8dfb-163ef9023375)


In the plot using the original features, there is some indication of clustering, particularly with a concentration of points at lower values of COUNT_HOUSE_BUILDING and TOTAL_CONSTRUCTION_VALUE. However, the structure is not very clear due to the spread and potential noise in the data.
In the PCA plot, there is a more distinct cluster of points near the origin, with some points spread out, indicating variability and the presence of outliers. This plot reveals a clearer structure, highlighting clusters and patterns that are less visible in the original feature plot.
PCA reduces the dimensionality by capturing the most significant variance, making patterns and clusters more apparent. In contrast, the original feature plot includes all variance, including noise, which can potentially obscure the structure. PCA also filters out less important details, focusing on the primary sources of variability. This leads to a clearer visualization of the inherent structure. The principal components are linear combinations of the original features, providing a new perspective that highlights the underlying structure more effectively.
After clustering, what observations belong to the same cluster?

 ![image](https://github.com/user-attachments/assets/c30444b4-ae9a-4907-b81f-3a04f6909f8a)

Data assigned to cluster 9
We analyzed data from cluster 1 and cluster 9. While cluster 1 had rows where ‘count’ variables such as Count_Apartment_Building and Count_House_Building were mostly 0s, Cluster 9 showed a pattern. As shown in the above table, data that belonged to cluster 9 shows some similarity if we look at the loan_total_value, savings_total and Construction_total_value. The have data points that are either close to one another or the same. Same can be seen with the ‘count’ variables, people who have a similar number of office buildings or house buildings were grouped together. 

One of the constraints encountered during hierarchical clustering was the Jupyter Notebook kernel crashing while running Python’s linkage library. The primary reason for the kernel crash was the large dataset size, which exceeded the computational capabilities of the environment.

To address this issue, one potential method could have been to reduce the dataset size by sampling. Sampling would involve selecting a representative subset of the data, which would decrease the computational load and make the linkage process more manageable. However, for this report, this approach has not been implemented.

Despite this limitation, the alternative clustering methods employed, such as k-means clustering, provided valuable insights into the data structure. Future analyses could consider implementing sampling or other dimensionality reduction techniques to enable successful hierarchical clustering with large datasets.

[Back to top](#table-of-contents)

---

## Conclusion
The application of unsupervised learning techniques to the Mongolian civil servants' declaration data has provided valuable insights into the dataset's structure. Principal Component Analysis (PCA) effectively reduced the dimensionality, preserving 81.64% of the total variance in just 16 components. K-means clustering further revealed three distinct groups within the data, highlighting patterns and potential areas of interest. These findings underscore the effectiveness of PCA and k-means clustering in handling and deriving meaningful insights from complex, multi-dimensional data. 

[Back to top](#table-of-contents)

---

## References
[1] Xacxom IAAC. (n.d.). Home. Xacxom IAAC. http://xacxom.iaac.mn/ (Retrieved 2024, May 26, from http://xacxom.iaac.mn/)

[2] Central Intelligence Agency. (n.d.). Mongolia. The World Factbook. https://www.cia.gov/the-world-factbook/countries/mongolia/ (Retrieved 2024, May 26, from https://www.cia.gov/the-world-factbook/countries/mongolia/)

[3] Independent Authority Against Corruption of Mongolia. (n.d.). Home. Independent Authority Against Corruption of Mongolia. https://iaac.mn/ (Retrieved 2024, May 26, from https://iaac.mn/)

[4] Selenium. (n.d.). Selenium with Python. Selenium Documentation. https://selenium-python.readthedocs.io/ (Retrieved 2024, June 2, from https://selenium-python.readthedocs.io/)

[5] OpenAI. (n.d.). Overview. OpenAI Platform. https://platform.openai.com/docs/overview (Retrieved 2024, June 2, from https://platform.openai.com/docs/overview)

[6] Mongolian Data Club. (n.d.). Home. Medium. https://dataclub.medium.com/ (Retrieved 2024, May 26, from https://dataclub.medium.com/)

[7] James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An Introduction to Statistical Learning with Applications in Python. (Original work published 2023) https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html

[Back to top](#table-of-contents)