import io
import re
import math
import Stemmer

filename="news_train.txt"
lines=[]
print("Importing data from input file", filename, "..")
with io.open(filename,encoding='utf-8') as file:
    for line in file:
        lines.append(line)
print("Finished importing.")

for i in range(len(lines)):
    lines[i]=lines[i].split('\t', maxsplit=1)
    lines[i][1].strip()

total_len = len(lines)
training_percent = 1
training_len = int(len(lines)*training_percent)
class_number = 10
class_prob = [["science", "style", "culture", "life", "economics",
               "business","travel", "forces", "media", "sport"],
              [0, 0, 0, 0, 0, #number of documents 
               0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0,# total number of words in documents
               0, 0, 0, 0, 0]]

word_list = {}
word_count = 0

filename="stopwords.txt"
stopwords=[]
with io.open(filename,encoding='utf-8') as file:
    for line in file:
        stopwords.append(re.sub('[^a-zа-я]', '', line))

stopwords = set(stopwords)

exists = 0

rusStemmer = Stemmer.Stemmer('russian')
engStemmer = Stemmer.Stemmer('english')

print("Training...")
for i in range(training_len):
    for j in range(class_number):
        if lines[i][0] == class_prob[0][j]:
            class_n = j
            class_prob[1][j] = class_prob[1][j] + 1
            temp = re.sub('[^a-zа-я\s]', '', lines[i][1].lower())
            temp = [w for w in temp.split() if w not in set(stopwords)]
            temp = ' '.join(temp)
            temp = re.split("\s+", temp)
            class_prob[2][j] = class_prob[2][j] + len(temp)
            for k in range(len(temp)):
                temp[k] = rusStemmer.stemWord(temp[k])
                key = temp[k]
                if key in word_list:
                    exists = 1
                if not exists:
                    word_list[key] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    word_list[key][j] = 1
                    word_count = word_count + 1
                else:
                    word_list[key][j] = int(word_list[key][j])+1
                exists = 0
            break
print("Finished training.")

print("Classifying...")
def classify(test_str):
    test_str = re.sub('[^a-zа-я\s]', '', test_str.lower())
    test_str = [w for w in test_str.split() if w not in set(stopwords)]
    test_str = ' '.join(test_str)
    test_words = re.split("\s+", test_str)
    test_prob = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(class_number):
        test_prob[i] = math.log(class_prob[1][i]/training_len, 2)
        for j in range(len(test_words)):
            temp = 0
            test_words[j] = rusStemmer.stemWord(test_words[j])
            val = word_list.get(test_words[j])
            if not val == None:
                temp = val[i]
            test_prob[i] = test_prob[i]+math.log((temp+1)/(len(word_list)+class_prob[2][i]), 2)
    for i in range(class_number):
        test_prob[i] = [test_prob[i], i]
    test_prob.sort()
    test_max_prob = class_prob[0][test_prob[9][1]]
    return test_max_prob
filename="news_test.txt"
test_news=[]
output = []
with io.open(filename,encoding='utf-8') as file:
    for line in file:
        test_news.append(line)
for i in range(len(test_news)):
    test_class = classify(test_news[i])
    output.append(test_class)
print("Finished classifying.")

print("Writing data to output file...")
with open("result.txt","w",newline="") as file:
    for i in range(len(test_news)):
        file.write(output[i])
        file.write("\n")
print("Finished.")

