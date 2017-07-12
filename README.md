Step 1: Preprocessing 

1. Get raw data in suitable format

2. Parsing data

Step 2: Training Distributed Semantic Representation (Word2Vec Model)

1. Use a Python Module called gensim.models.word2vec for this.

2. Train a model using only the sentences from the corpus.

3. This generates vectors for all the words in the corpus.

4. This model can now be used to get vectors for the words.

5. For unknown words, we use the vectors of words with frequency one.

Step 3: Make Word Vectors 

Step 4: Training For Machine Learning Scores

1. Use training data to make the models for:

a. Support Vector Machines - Scikit Learn Python

b. Multi Layer Perceptron Neural Network - Scikit Learn Python

c. Naive Bayes Classifier - Scikit Learn Python

d. Decision Tree Classifier - Scikit Learn Python

e. Random Forest Classifier - Scikit Learn Python

f. Logistic Regression Classifier - Scikit Learn Python

g. Recurrent Neural Networks - PyBrain module Python

2. Classify the test data using the models.

3. Compare the output results with the actual classes to get the accuracy.