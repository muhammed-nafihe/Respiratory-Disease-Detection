# Data Mining and Machine Learning Group Coursework
# Respiratory Disease Detection
## Group Members

> [!IMPORTANT]
1.TIJO THOMAS <br />
2.VISHNUDEV AMMUPPILLY  <br />
3.MUHAMMED NAFIHE (H00446008) <br />
4.MARIYA SEBASTIAN<br />

## Initial Project Proposal
* Our Group Name : NTVM
* Project Title : Respiratory Disease Detection
* To check whether a person is effected  with a respiratory  diseases or not </br >
* Our project plan is to detect and identify different respiratory diseases mainly COPD and COVID 18 from the respiratory sounds recorded by an electrical stethoscope. After the 
  pandemic wave, people all over the world are susceptible more to respiratory diseases and the detection of such disease in early stages is necessary for better health care. Our team 
  is also planning to study further into the way the respiratory system works in humans and the data available on it, and how can we better obtain useful methodologies for different 
  studies. <br />
* In this project, we are planning to build a model that can identify the diseases from a breathing sound.

### Research objectives

> [!NOTE]
> What are the questions you are trying to answer? What are the goals of your project?  <br />
>* Build a model to classify respiratory diseases  <br />
>* Is the breathing sound same for people with and without respiratory diseases? IF no, what are the variations shown for that in a graph?  <br />
>* How to digitalize the difference between normal person's breathing sound and a Respriratory disease infected patient's breathing sound ?  <br />
>* Build a model to detect if a recording contains crackles, wheezes or both.  <br />
> 
> 

### Milestones
  
  1. week 1 ->  week 2 -> Formed a group of four members. <br />
  2. week 3 -> Discussed various topics and available datasets. we went through a lot of datasets available on different websites. Selected a dataset for 
               Respiratory Sound Analysis. we also planed some works for next week such as making a better plan to work out our 
               project perfectly and to go through the dataset which is selected. <br />
  3. week 4 -> Discussion on project plan and data exploration.<br />
  4. week 5 -> Data description and visualization started <br />
  5. week 6 -> Data description and visualization completed and  Data preprocessing started</br> 
  6. week 7 -> Data preprocessing completed and look into different clustering method which suites for out dataset <br />
  7. week 8 -> We have started training the dataset with different types clustering Algoritham<br />
  8. week 9 -> Completed K-means clustering and started Decision tree. We had decied to do classification decision tree on this data set because our target variables are categorical.
  9. week10 -> Done with classification Decision tree and achieved a accuracy of 71% and and started with CNN<br />
  10. week 11 -> Completed CNN and got an accuracy of 94.<br/>


## Findings Report

<!-- Below you should report all of your findings in each section. You can fill this out as the project progresses. -->


### Datasets
1. Respiratory Sound Database - https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
#### Dataset description
<!-- Briefly describe your task and dataset -->
 A person's Respiratory sounds are the most important indicator to tell whether a person is effected  by any respiratory diseases or not. The sound of a person's breathing  is directly related to air movement, changes within lung tissue, and the position of secretions within the lung. which is a crucial  and simple way to find a person effected  by respiratory diseases.  <br />
   This dataset contains the recorded breathing sounds of normal people and persons infected with COPD. The data includes both clean respiratory sounds as well as noisy recordings that simulate real-life conditions.

#### Dataset examples
<!-- Add a couple of example instances and the dataset format -->
Example of dataset:<br />
 0  0.036	0.579	0	0	101	sc	101_1b1_Al_sc_Meditron<br />  
 1  0.579	2.450	0	0	101	sc	101_1b1_Al_sc_Meditron<br />
 2  2.450	3.893	0	0	101	sc	101_1b1_Al_sc_Meditron<br />

Elements contained in the filenames:<br />
1. Patient number (101,102,...,226)<br />
2. Recording index<br />
3. Chest location (Trachea (Tc), {Anterior (A), Posterior (P), Lateral (L)}{left (l), right (r)})<br />
4. Acquisition mode (sequential/single channel (sc), simultaneous/multichannel (mc))<br />
5. Recording equipment (AKG C417L Microphone, 3M Littmann Classic II SE Stethoscope, 3M Litmmann 3200 Electronic Stethoscope, WelchAllyn 
   Meditron Master Elite Electronic Stethoscope)


#### Dataset exploration
<!-- What is the size of the dataset? --> 
Size of dataet is 4 GB with a total of 5.5 hours of recordings containing 6898 respiratory cycles - 1864 contain crackles, 886 contain wheezes and 506 contain both crackles and wheezes.

* We have extracted audio features from audio files and made a CSV file to use as our dataset.
* Link to the histogram of number of instances for each class is :
     (https://github.com/dmml-heriot-watt/group-coursework-ntvm/assets/106021255/9641363f-2bec-4fdb-bce4-f7538772c740)
          COPD              5746
          Healthy            322
          Pneumonia          285
          URTI               220
          Bronchiolitis      160
          Bronchiectasis     104
          LRTI                32
          Asthma               6
          Name: diagnosis, dtype: int64
* This is the frequency of each classes in our dataset. Here the majority class is COPD with 5746 instances and all other intances 
   have only less than 350 number of instances. From thistable we can understood that our dataset is iunnbalanced.

* Link to scatter plot visualization of our dataset :
                https://github.com/dmml-heriot-watt/group-coursework-ntvm/assets/106021255/2f94d450-e558-40dc-902c-18a747817189)
* Link to Box plot:
                (https://github.com/dmml-heriot-watt/group-coursework-ntvm/assets/106021255/b2138ace-baf1-4f0c-9704-0fd96bf88683)


<!-- Analysis of your dataset -->

### Clustering
We have used K-means clustering algorithm in the machine learning he process involves importing necessary libraries, loading and possibly preprocessing a dataset, applying the K-Means algorithm to identify clusters, and finally visualizing these clusters.

Advantages: Simple and computationally efficient. Works well when clusters are spherical and have a similar size.

#### Experimental design
As we have supervised the dataset the dataset is already labeled as both input features and corresponding output labels are already inside. Our data set is an imbalanced data set in such a way that instances of COPD are too high compared to other labels. So we have done both undersampling and made the data set in such a way that all that data became balanced in the process we had to remove diseases like 'LRTI', and 'Asthma' because the data containing the two diseases was very low. the modified and balanced data are given below

Diagnosis

COPD              104 </br>
Bronchiolitis     104 </br>
Bronchiectasis    104 </br>
Pneumonia         104 </br>
URTI              104 </br>
Healthy           104 </br>

After balancing the data set  we have applied PCA (Principal components analysis) to reduce to 2 componets for visualization and to find the optimal number of cluster we have used the Elbow method  after ploting the elbow method we have Assumed the optimal number of clusters from the elbow plot is 3 (we have adjusted based on our plot)

Visualizing the clusters  withe 3 clusturs  and 3 centroid. The graph likely displays the clustering results. Points are colored differently to represent each cluster.and the 3 clusters represent a grouping of data points that are similar to each other based on the features used in the dataset. The centroids of these clusters might also be plotted, usually represented in a contrasting color.
#### Results
<!-- Tables showing the results of your experiments -->

Data points in Cluster 1

Balanced data

diagnosis
COPD              104 </br>
Bronchiolitis     104 </br>
Bronchiectasis    104</br>
Pneumonia         104</br>
URTI              104</br>
Healthy           104</br>

K-Means Clustering

Index	  feature_0	  feature_1	...	feature_22	Cluster
0	     1.764052	    0.400157	...	0.864436	     0
1	     0.742165	    2.269755	...	-0.438074	     1
2	     1.252795   	0.777490	...	-0.907298      1
3	     0.051945    	0.729091	...	1.222445	     1
4    	0.208275	    0.976639	...	-0.268003	     1

Data points in Cluster 2

Index	feature_0	 feature_1...	feature_22	Cluster
7	    1.929532	 0.949421	...	2.223403	    2
11	  0.319328	 0.691539	...	0.390953	    2
15	  0.280355	 0.364694	...	0.394850	    2
19	  0.005293	 0.800565	... 0.050604	    2
23	  0.864052	 2.239604	... 0.096321	    2

Data points in Cluster 3

Index	 feature_0	feature_1	...	feature_22	Cluster
1	     0.742165	   2.269755	...	-0.438074	    3
2	     1.252795	   0.777490	...	-0.907298	    3
3	     0.051945	   0.729091	...	1.222445	    3
4	     0.208275	   0.976639	...	-0.268003	    3
5	     0.802456	   0.947252	...	-0.208299	    3

Each row in these tables represents a data point, with its features (feature_0 to feature_22) and the cluster number ( 1,2 and 3) to which it has been assigned by the K-Means algorithm. These mappings help in understanding the characteristics and distribution of data points within each cluster. ​

#### Discussion
<!-- A brief discussion on the results of your experiment -->
The process involves importing necessary libraries, loading and possibly balancing the dataset, applying the K-Means algorithm to identify clusters, and finally visualizing these clusters by elbow method we got as 3 . The visualization likely shows the data points grouped into three distinct clusters and three centriods  each representing a grouping of similar data points. 

### Decision Trees

#### Experimental design
<!-- Describe your experimental design and choices for the week. -->
* Our data set have 26 input features and 8 target labels. such as Asthma, URTI, Pneumonia, COPD, Healthy, LRTI, Bronchiectasis, Bronchiolitis.
* We used Classification Decision Tree algorithm.
* Our data set is a imbalanced data set such that instances of COPD is too high compared to other labels. So we have done both undersampling and oversampling. But we choosed 
  oversampling over undersampling based on accuracy.
* After over-sampling counts of our target variables are:
          COPD                     4017
          Healthy                  4017
          Pneumonia                4017
          Bronchiolitis            4017
          Bronchiectasis           4017
          URTI                     4017
          LRTI                     4017
          Asthma                   4017
          Name: diagnosis, dtype: int64

* We also used GridSearchCV to do hyperparameter tuning. Parameters we used in GridSearchCV are criterion, max_depth, min_samples_split, min_samples_leaf. Also printed the best value 
  for each parameter.
* Also done codes for finding accuracy of model, classification report and confusion matrix.
#### Results
* Link to Decision tree:
            (https://github.com/dmml-heriot-watt/group-coursework-ntvm/assets/106021255/e41aa13f-26a9-4bda-bfba-2b7faa892a8d)
* Classification Report:<br />
                         precision    recall  f1-score   support
          
                  Asthma       1.00      0.00      0.00         3
          Bronchiectasis       0.32      0.82      0.46        38
           Bronchiolitis       0.13      0.56      0.22        50
                    COPD       0.97      0.78      0.87      1729
                 Healthy       1.00      0.00      0.00        89
                    LRTI       0.19      0.64      0.29        11
               Pneumonia       0.19      0.68      0.30        77
                    URTI       0.07      0.08      0.07        66
          
                accuracy                           0.71      2063
               macro avg       0.49      0.44      0.28      2063
            weighted avg       0.88      0.71      0.75      2063
* Accuracy: 0.7130392632089191
* This is the classification report and accuracy of our decision tree model. Our model had achieved 71% of Accuracy, it meanas that 71% of predictions are correct.
* This tables gives the idea about how well the decision tree model works on different classes of test data set.
* Link to Confussion Matrix Heat Map:
    (https://github.com/dmml-heriot-watt/group-coursework-ntvm/assets/106021255/3be4e7a1-8179-47a2-afe9-32fcf7201ed4)<br />
*Confussion Matrix :<br />
                 [[     0       0       0       3       0       0       0       0]<br />
                 [      0      31       3       0       0       0       0       4]<br />
                 [      0       6      28       2       0       2      11       1]<br />
                 [      0      56     108    1348       0      15     143      59]<br />
                 [      0       3      30      10       0       3      43       0]<br />
                 [      0       0       0       0       0       7       4       0]<br />
                 [      0       0      12      11       0       2      52       0]<br />
                 [      0       1      27       9       0       8      16       5]]<br />
* In this confussion matrix horizontal lines represents the predicted classes, and the vertical represents the actual classes.
* It also explains for each class how many predition have got correct and how many got worng. When we looking in to column number 4 and row numbber 4, it relates to class COPD. COPD
  class have 1348 correct Predictions.
#### Discussion
* This decision tree model is highly effective for class like COPD which have the high precision, recall, and F1-score. But n the case of classes like Asthma and Healthy the model have 
  failed to make more number number of correct predictions. This may due to the high number of instances of COPD in test data set and lower number of instances of Healthy and Asthma.
* Hyperparameter tuning is done with the help of GridSearchCV. We had 162 totals fits for our decision tree. Out of 162 models we 
  selected a model of parameters criterion as entropy, maximum depth of 4, min_samples_leaf of 5, min_samples_split of 5.
* Link of notebook : http://localhost:8888/notebooks/DM%26ML/decision_tree_oversampling.ipynb#
### Neural Networks
Input <br />
The input features consist of three types: MFCCs, Chroma, and Mel Spectrogram.<br />
Mfcc : short term power spectrum of the sound<br />
Mel spec changes relative to regular spectrogram<br />
Chroma for analysing pitch<br />
Output<br />
To find whether the data sound is classifies correctly as COPD, healthy <br />
#### Experimental design<br />
<!-- Describe your experimental design and choices for the week. -->
•	Our CNN neural network architecture comprises three sub models for MFCC, Chroma, and Mel Spectrogram<br />
•	Each sub model consists of convolutional layers with different kernels, batch normalization, activation functions, and max-pooling layers<br />
•	The final part concatenates the outputs of these sub models and includes dense layers with dropout<br />
#### Results
As we used 3 different inputs to get the final% model with different layers in neural network creation we got better accuracy than we expected and it can be thought realistically feasible too with "loss: 0.1831 - accuracy: 0.9443"( shown in the notebooks code) and the trial with changing some parameters gave above 90% too <br/>

<!-- Tables showing the results of your experiments -->
The loss function we used is sparse categorical cross entropy as it is useful for integer-encoded labels and we used a learning rate of .001. Used the train and validation data for monitoring the model's performance during training <br />


#### Discussion
<!-- A brief discussion on the results of your experiment -->
From the results we can conclude that our model is good fit for this dataset and the classification for it ,from trial a lower learning rate is found to better. <br>
We have learned how to do CNN models and how to construct more layers and also how to do the audio classification with CNN<br>

### Conclusion
<!-- Final conclusions regarding your initial objectives -->
* We have done data visualization by using Scatter plots, box plots and histograms. From the histogram, we concluded that COPD have a 
  higher number of instances nearly 5000 and all other classes have only less number. In the preprocessing, we had checked for any null 
  value in the data.Then we have extracted 26 audio features and created a CSV file to use for all analysis. Then we have merged this 
  data with patient diagnosis data which includes patient id and corresponding disease.

* The  K-means Clustring process involves importing necessary libraries, loading and possibly balancing the dataset, applying the K- 
  Means algorithm to identify clusters, and finally visualizing these clusters by elbow method we got as 3 . The visualization likely 
  shows the data points grouped into three distinct clusters and three centriods  each representing a grouping of similar data points.
* In the decision tree, we have done a classification-type decision tree model and got 71% of correct class predictions. The classes 
  with the high number of instances such as COPD got a good rate of correct prediction, but classes like Asthma, healthy etc. got only 
  a smaller number of correct predictions. We used a classification report and confusion matrix to evaluate our model.

* The current model is a good fit for CNN classification for the used dataset. the model can be further used for all the disease and we found the accuracy with 
  sparse cartegorical cross entropy to be above 90%.
* We have good accuracy in all models and in future works we can try to iinclude more diseases and inmproved datset
