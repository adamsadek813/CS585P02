import sys
import pandas as pd
import string
import math

# ask user for TRAIN_SIZE input
# check if input is valid and only one input
if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) < 20 or int(sys.argv[1]) > 80:
    TRAIN_SIZE = 80
else:
    TRAIN_SIZE = int(sys.argv[1])





data_pre = pd.read_csv("bumble_google_play_reviews.csv")
data = data_pre.dropna(subset=['content']).reset_index(drop=True)
data['review_class'] = data['score'].apply(lambda x: 'positive' if x > 2 else 'negative')


# 80-20 split
test_index = int(len(data) * 0.8)
train_index = int(len(data) * float(TRAIN_SIZE / 100.0))
train_data = data.iloc[:train_index]
test_data = data.iloc[test_index:]

unique_vocabulary = []
prior_probabilities = {}
translation_table = []
stop_words = []
num_pos_docs_including_word = []
num_neg_docs_including_word = []
pos_indices = []
neg_indices = []


### Naive Bayes without stop words

def naive_bayes_train_no_stop_words(train_data):
    global unique_vocabulary
    global prior_probabilities
    global translation_table
    global stop_words
    global num_pos_docs_including_word
    global num_neg_docs_including_word
    global pos_indices
    global neg_indices
    
    for x in range(len(train_data)):
        if train_data['review_class'][x] == 'positive':
            pos_indices.append(x)
        if train_data['review_class'][x] == 'negative':
            neg_indices.append(x)
    
    prior_probabilities = {
        'positive': len(train_data[train_data['review_class'] == 'positive']) / len(train_data),
        'negative': len(train_data[train_data['review_class'] == 'negative']) / len(train_data)
    }
    
    vocabulary = []
    # list of common stop words
    stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with']


    translation_table = str.maketrans('', '', string.punctuation)

    for x in range(len(data['content'])):
        content = data['content'][x]
        lowercase_content = content.lower()
        content_no_punct = lowercase_content.translate(translation_table)
        vocabulary.extend(content_no_punct.split())



    for item in vocabulary:
        if item not in unique_vocabulary and item not in stop_words:
            unique_vocabulary.append(item)

    binary_bag = [[0]*len(unique_vocabulary) for _ in range(len(train_data))]

    for x in range(len(train_data)):
        for y in range(len(train_data['content'][x].split())):
            words_in_one_sentence = (train_data['content'][x].split()[y]).lower().translate(translation_table)
            if words_in_one_sentence != '' and words_in_one_sentence not in stop_words:   
                index_of_word = unique_vocabulary.index(words_in_one_sentence)
                binary_bag[x][index_of_word] = 1

    # sigma(z) = 1 / (1 + e^(-z))

    num_docs_including_word = [0]*len(unique_vocabulary)
    num_pos_docs_including_word = [0]*len(unique_vocabulary)
    num_neg_docs_including_word = [0]*len(unique_vocabulary)

    for word in range(len(unique_vocabulary)):
        for document in range(len(binary_bag)):
            if binary_bag[document][word] == 1:
                num_docs_including_word[word] += 1
                if document in pos_indices:
                    num_pos_docs_including_word[word] += 1
                if document in neg_indices:
                    num_neg_docs_including_word[word] += 1
    
    prob_word_given_pos_class = []
    prob_word_given_neg_class = []

    count_pos_class = math.fsum(num_pos_docs_including_word)
    count_neg_class = math.fsum(num_neg_docs_including_word)


    for x in range(len(unique_vocabulary)):
        prob_word_given_pos_class.append(float(num_pos_docs_including_word[x] + 1) / float(count_pos_class + len(unique_vocabulary)))
        prob_word_given_neg_class.append(float(num_neg_docs_including_word[x] + 1) / float(count_neg_class + len(unique_vocabulary)))        
    
    
test_data = test_data.reset_index(drop=True)

def naive_bayes_classifier_no_stop_words(test_data):
    predictions = []
    # number of unique words in the vocabulary
    V = len(unique_vocabulary)
    # smoothing factor
    alpha = 1
    
    for x in range(len(test_data)):
        pos_prob = math.log(prior_probabilities['positive'])
        neg_prob = math.log(prior_probabilities['negative'])
        document_words = test_data['content'][x].split()

        for word in document_words:
            word = word.lower().translate(translation_table)
            if word != '' and word not in stop_words:
                if word in unique_vocabulary:
                    index_of_word = unique_vocabulary.index(word)
                    # Calculate word probability with smoothing
                    word_pos_prob = (num_pos_docs_including_word[index_of_word] + alpha) / (len(pos_indices) + alpha * V)
                    word_neg_prob = (num_neg_docs_including_word[index_of_word] + alpha) / (len(neg_indices) + alpha * V)
                else:
                    # Apply smoothing for words not in the vocabulary
                    word_pos_prob = alpha / (len(pos_indices) + alpha * V)
                    word_neg_prob = alpha / (len(neg_indices) + alpha * V)
                
                pos_prob += math.log(word_pos_prob)
                neg_prob += math.log(word_neg_prob)
                
                # convert to linear space so that we can compare the probabilities (add up to 1)
                pos_prob = 10**pos_prob
                neg_prob = 10**neg_prob
                
        if pos_prob >= neg_prob:
            predictions.append('positive')
        else:
            predictions.append('negative')
    return predictions

# predictor for a particular review
def naive_bayes_individual_no_stop_words(review):
    # number of unique words in the vocabulary
    V = len(unique_vocabulary)
    # alpha is the smoothing factor
    alpha = 1
    
    pos_prob = math.log(prior_probabilities['positive'])
    neg_prob = math.log(prior_probabilities['negative'])
    document_words = review.split()

    for word in document_words:
        word = word.lower().translate(translation_table)
        if word != '' and word not in stop_words: 
            if word in unique_vocabulary:
                index_of_word = unique_vocabulary.index(word)
                word_pos_prob = (num_pos_docs_including_word[index_of_word] + alpha) / (len(pos_indices) + V)
                word_neg_prob = (num_neg_docs_including_word[index_of_word] + alpha) / (len(neg_indices) + V)
            else:
                word_pos_prob = 1 / (len(pos_indices) + V)
                word_neg_prob = 1 / (len(neg_indices) + V)
            
            pos_prob += math.log(word_pos_prob)
            neg_prob += math.log(word_neg_prob)
            
            # convert to linear space so that we can compare the probabilities (add up to 1)
            pos_prob = 10**pos_prob
            neg_prob = 10**neg_prob

    # if pos_prob >= neg_prob:
    #     return 'positive'
    # else:
    #     return 'negative'
    return pos_prob, neg_prob
    

### Naive Bayes with stop words

def naive_bayes_train(train_data):
    global unique_vocabulary
    global prior_probabilities
    global num_pos_docs_including_word
    global num_neg_docs_including_word
    global pos_indices
    global neg_indices
    
    for x in range(len(train_data)):
        if train_data['review_class'][x] == 'positive':
            pos_indices.append(x)
        if train_data['review_class'][x] == 'negative':
            neg_indices.append(x)
    
    prior_probabilities = {
        'positive': len(train_data[train_data['review_class'] == 'positive']) / len(train_data),
        'negative': len(train_data[train_data['review_class'] == 'negative']) / len(train_data)
    }
    
    vocabulary = []

    for x in range(len(data['content'])):
        content = data['content'][x]
        vocabulary.extend(content.split())



    for item in vocabulary:
        if item not in unique_vocabulary:
            unique_vocabulary.append(item)

    binary_bag = [[0]*len(unique_vocabulary) for _ in range(len(train_data))]

    for x in range(len(train_data)):
        for y in range(len(train_data['content'][x].split())):
            words_in_one_sentence = (train_data['content'][x].split()[y])
            if words_in_one_sentence != '':   
                index_of_word = unique_vocabulary.index(words_in_one_sentence)
                binary_bag[x][index_of_word] = 1

    num_docs_including_word = [0]*len(unique_vocabulary)
    num_pos_docs_including_word = [0]*len(unique_vocabulary)
    num_neg_docs_including_word = [0]*len(unique_vocabulary)

    for word in range(len(unique_vocabulary)):
        for document in range(len(binary_bag)):
            if binary_bag[document][word] == 1:
                num_docs_including_word[word] += 1
                if document in pos_indices:
                    num_pos_docs_including_word[word] += 1
                if document in neg_indices:
                    num_neg_docs_including_word[word] += 1
    
    prob_word_given_pos_class = []
    prob_word_given_neg_class = []

    count_pos_class = math.fsum(num_pos_docs_including_word)
    count_neg_class = math.fsum(num_neg_docs_including_word)


    for x in range(len(unique_vocabulary)):
        prob_word_given_pos_class.append(float(num_pos_docs_including_word[x] + 1) / float(count_pos_class + len(unique_vocabulary)))
        prob_word_given_neg_class.append(float(num_neg_docs_including_word[x] + 1) / float(count_neg_class + len(unique_vocabulary)))        
    
    
test_data = test_data.reset_index(drop=True)

def naive_bayes_classifier(test_data):
    predictions = []
    # number of unique words in the vocabulary
    V = len(unique_vocabulary)
    # smoothing factor
    alpha = 1
    
    for x in range(len(test_data)):
        pos_prob = math.log(prior_probabilities['positive'])
        neg_prob = math.log(prior_probabilities['negative'])
        document_words = test_data['content'][x].split()

        for word in document_words:
            if word != '':
                if word in unique_vocabulary:
                    index_of_word = unique_vocabulary.index(word)
                    # Calculate word probability with smoothing
                    word_pos_prob = (num_pos_docs_including_word[index_of_word] + alpha) / (len(pos_indices) + alpha * V)
                    word_neg_prob = (num_neg_docs_including_word[index_of_word] + alpha) / (len(neg_indices) + alpha * V)
                else:
                    # Apply smoothing for words not in the vocabulary
                    word_pos_prob = alpha / (len(pos_indices) + alpha * V)
                    word_neg_prob = alpha / (len(neg_indices) + alpha * V)
                
                pos_prob += math.log(word_pos_prob)
                neg_prob += math.log(word_neg_prob)
                
                # convert to linear space so that we can compare the probabilities (add up to 1)
                pos_prob = 10**pos_prob
                neg_prob = 10**neg_prob
                
        if pos_prob >= neg_prob:
            predictions.append('positive')
        else:
            predictions.append('negative')
    return predictions

# predictor for a particular review
def naive_bayes_individual(review):
    # number of unique words in the vocabulary
    V = len(unique_vocabulary)
    # alpha is the smoothing factor
    alpha = 1
    
    pos_prob = math.log(prior_probabilities['positive'])
    neg_prob = math.log(prior_probabilities['negative'])
    document_words = review.split()

    for word in document_words:
        if word != '': 
            if word in unique_vocabulary:
                index_of_word = unique_vocabulary.index(word)
                word_pos_prob = (num_pos_docs_including_word[index_of_word] + alpha) / (len(pos_indices) + V)
                word_neg_prob = (num_neg_docs_including_word[index_of_word] + alpha) / (len(neg_indices) + V)
            else:
                word_pos_prob = 1 / (len(pos_indices) + V)
                word_neg_prob = 1 / (len(neg_indices) + V)
            
            pos_prob += math.log(word_pos_prob)
            neg_prob += math.log(word_neg_prob)
            
            # convert to linear space so that we can compare the probabilities (add up to 1)
            pos_prob = 10**pos_prob
            neg_prob = 10**neg_prob

    # if pos_prob >= neg_prob:
    #     return 'positive'
    # else:
    #     return 'negative'
    return pos_prob, neg_prob

### METRICS

def calculate_metrics(predictions, actual_labels):
    # Initialize counters
    TP = FP = TN = FN = 0
    
    # Count occurrences
    for pred, actual in zip(predictions, actual_labels):
        if pred == 'positive' and actual == 'positive':
            TP += 1
        elif pred == 'positive' and actual == 'negative':
            FP += 1
        elif pred == 'negative' and actual == 'positive':
            FN += 1
        elif pred == 'negative' and actual == 'negative':
            TN += 1
    
    # Calculate metrics
    sensitivity = TP / (TP + FN) if TP + FN else 0  
    specificity = TN / (TN + FP) if TN + FP else 0
    precision = TP / (TP + FP) if TP + FP else 0
    negative_predictive_value = TN / (TN + FN) if TN + FN else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN else 0
    F_score = (2 * precision * sensitivity) / (precision + sensitivity) if precision + sensitivity else 0

    return {
        'Number of true positives': TP,
        'Number of true negatives': TN,
        'Number of false positives': FP,
        'Number of false negatives': FN,
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Negative Predictive Value': negative_predictive_value,
        'Accuracy': accuracy,
        'F-score': F_score
    }
    
    
print("Petkov, Kamen, A20464521 solution:")
print(f"Training set size: {TRAIN_SIZE}%\n")

# print("Training classifier...")
# ### Predictions no stop words
# # train
# naive_bayes_train_no_stop_words(train_data)

# print("Testing classifier...")
# print("Test results / metrics:")
# # extract actual labels
# actual_labels = test_data['review_class'].tolist()


# # get predictions
# predictions_no_stop_words = naive_bayes_classifier_no_stop_words(test_data)

# metrics_no_stop_words = calculate_metrics(predictions_no_stop_words, actual_labels)
# for metric, value in metrics_no_stop_words.items():
#     print(f"{metric}: {value}")



    
# # ask user for input of a sentence
# sentence = input("Enter your sentence: ")
# classified_as = ""
# # display classifier decision along with P(CLASS_A |S) and P(CLASS_B | S) values on screen
# pos_prob_no_stop_words, neg_prob_no_stop_words = naive_bayes_individual_no_stop_words(sentence)
# if pos_prob_no_stop_words >= neg_prob_no_stop_words:
#     classified_as = "positive"
# else:
#     classified_as = "negative"
# print("Sentence S: \n", sentence, "\nwas classified as: ", classified_as)


# print(f"P(Positive | S): {pos_prob_no_stop_words}")   
# print(f"P(Negative | S): {neg_prob_no_stop_words}")
    
    
    
    
    
    
### Predictions with no improved naive bayes
# train
print("Training classifier...")
naive_bayes_train(train_data)

print("Testing classifier...")
print("Test results / metrics:\n")
# extract actual labels
actual_labels = test_data['review_class'].tolist()


# get predictions
predictions = naive_bayes_classifier(test_data)

metrics= calculate_metrics(predictions, actual_labels)
for metric, value in metrics.items():
    print(f"{metric}: {value}")



    
# ask user for input of a sentence
sentence = input("\n\nEnter your sentence: \n\n")
classified_as = ""
# display classifier decision along with P(CLASS_A |S) and P(CLASS_B | S) values on screen
pos_prob, neg_prob = naive_bayes_individual(sentence)
if pos_prob >= neg_prob:
    classified_as = "positive"
else:
    classified_as = "negative"
print("\nSentence S: \n\n", sentence, "\n\nwas classified as: ", classified_as)


print(f"P(Positive | S): {pos_prob}")   
print(f"P(Negative | S): {neg_prob}")








new_sentence_bool = input("Do you want to enter another sentece [Y/N]?")
while new_sentence_bool.lower() == 'y':
    sentence = input("Enter your sentence: ")
    classified_as = ""
    # display classifier decision along with P(CLASS_A |S) and P(CLASS_B | S) values on screen
    pos_prob, neg_prob = naive_bayes_individual(sentence)
    if pos_prob >= neg_prob:
        classified_as = "positive"
    else:
        classified_as = "negative"
    print("Sentence S: \n", sentence, "\nwas classified as: ", classified_as)
    print(f"P(Positive | S): {pos_prob}")
    print(f"P(Negative | S): {neg_prob}")
    new_sentence_bool = input("Do you want to enter another sentece [Y/N]?")
    






# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc

# actual_labels = [1 if label == 'positive' else 0 for label in test_data['review_class']]

# # Calculate probabilities for each model
# probabilities_no_stop_words = [naive_bayes_individual_no_stop_words(text)[0] for text in test_data['content']]
# probabilities = [naive_bayes_individual(text)[0] for text in test_data['content']]

# # Compute ROC curve and ROC area for each model
# fpr_no_stop_words, tpr_no_stop_words, _ = roc_curve(actual_labels, probabilities_no_stop_words)
# roc_auc_no_stop_words = auc(fpr_no_stop_words, tpr_no_stop_words)

# fpr, tpr, _ = roc_curve(actual_labels, probabilities)
# roc_auc = auc(fpr, tpr)

# # Plotting
# plt.figure()
# lw = 2
# plt.plot(fpr_no_stop_words, tpr_no_stop_words, color='darkorange', lw=lw, label='Strict Model (area = %0.2f)' % roc_auc_no_stop_words)
# plt.plot(fpr, tpr, color='darkblue', lw=lw, label='Normal (Untouched) Model (area = %0.2f)' % roc_auc)

# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic Comparison')
# plt.legend(loc="lower right")
# plt.show()

# print("ROC curve for the model without stop words:")
# print(f"AUC: {roc_auc_no_stop_words:.2f}")
# print("ROC curve for the model with stop words:")
# print(f"AUC: {roc_auc:.2f}")