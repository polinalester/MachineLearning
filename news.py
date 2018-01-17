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
print("Finished importing.")

#stopwords = re.sub('[^a-zа-я]', '', stopwords)
stopwords = set(stopwords)
#stopwords = set(get_stop_words('ru'))
#stopwords=set(['бы', 'а', 'в', 'во', 'вот', 'для', 'до',
#                 'если', 'же', 'за', 'и', 'из', 'или', 'к',
#                 'ко', 'между', 'на', 'над', 'но', 'о', 'об',
#                 'около', 'от', 'по', 'под', 'при', 'про',
#                 'с', 'среди', 'то', 'у', 'чтобы'])
#print(stopwords)
#stopwords = set(nltk.corpus.stopwords.words('russian'))
#print(stopwords)

exists = 0

rusStemmer = Stemmer.Stemmer('russian')
engStemmer = Stemmer.Stemmer('english')

print("Training...")
for i in range(training_len):
    for j in range(class_number):
        if lines[i][0] == class_prob[0][j]:
            class_n = j
            class_prob[1][j] = class_prob[1][j] + 1
            #temp = lines[i][1].lower().replace('[^a-z\s]', '').split()
            temp = re.sub('[^a-zа-я\s]', '', lines[i][1].lower())
            temp = [w for w in temp.split() if w not in set(stopwords)]
            temp = ' '.join(temp)
            temp = re.split("\s+", temp)
            #print(temp)
            #temp=temp.split()
            #temp = re.split("[, \!?:\t\n—().«»]+", lines[i][1])
            class_prob[2][j] = class_prob[2][j] + len(temp)
            for k in range(len(temp)):
                temp[k] = rusStemmer.stemWord(temp[k])
                key = temp[k]
                #print(temp[k])
                #class_prob[2][j] = class_prob[2][j] + len(temp)
                if key in word_list:
                    exists = 1
                if not exists:
                    word_list[key] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    word_list[key][j] = 1
                    word_count = word_count + 1
                else:
                    word_list[key][j] = int(word_list[key][j])+1
                exists = 0
                #print(word_list[word_count-1])
            break
print("Finished training.")

#indices = list(range(0, len(word_list)))
#indices = set(indices)
#for i in range(len(word_list)):
#    for j in range(len(word_list)):
#        if word_list[]


print("Classifying.")

def classify(test_str):
    #test_str = 'Ученые доказали наличие индивидуальности у акул	Австралийские ученые впервые обнаружили у акул индивидуальные особенности характера. Об этом сообщается в журнале Journal of Fish Biology.Изучая рыб, обитающих у восточного побережья Австралии, биологи обнаружили у акул четкие и устойчивые реакции на непривычную среду и стресс. По их мнению, именно стабильность и предсказуемость поведенческих реакций и определяет характер живого существа.Для определения уровня храбрости отдельных особей (их склонности идти на риск) ученые разработали специальные эксперименты. Сначала акул переселили в новый аквариум и подсчитали, сколько времени каждой особи потребуется на то, чтобы вылезти из «домика» (специального убежища) и посмотреть на новую среду. Во втором эксперименте акул хватали и пытались вытащить на сушу, а потом снова бросали в аквариум и смотрели, насколько быстро они приходят в себя.Выяснилось, что в ходе экспериментов поведение каждой особи было примерно одинаковым. То есть отдельные акулы вели себя храбрее остальных.«Эти результаты удивительные — они показали, что акулы — это не просто холодные машины. Акула, как и человек, является личностью со своими привычками и особенностями поведения», — отметил соавтор статьи Калам Браун (Culum Brown).В 2015 году в ходе сходного эксперимента бельгийские ученые нашли у тараканов индивидуальные особенности поведения: особи разделились на боязливых (готовых всегда спрятаться в укрытие) и смелых авантюристов, идущих на риск в поисках добычи.#test_words = re.split("[, \-!?:\t]+", test_str.lower())
    test_str = re.sub('[^a-zа-я\s]', '', test_str.lower())
    test_str = [w for w in test_str.split() if w not in set(stopwords)]
    test_str = ' '.join(test_str)
    test_words = re.split("\s+", test_str)
    test_prob = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #test_prob = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
    #for i in range(len(test_prob)):
        #print(class_prob[0][test_prob[i][1]])
    #print(test_prob)
    test_max_prob = class_prob[0][test_prob[9][1]]
    return test_max_prob
    #print("Class: ", test_max_prob)

#test = 'Ученые доказали наличие индивидуальности у акул	Австралийские ученые впервые обнаружили у акул индивидуальные особенности характера. Об этом сообщается в журнале Journal of Fish Biology.Изучая рыб, обитающих у восточного побережья Австралии, биологи обнаружили у акул четкие и устойчивые реакции на непривычную среду и стресс. По их мнению, именно стабильность и предсказуемость поведенческих реакций и определяет характер живого существа.Для определения уровня храбрости отдельных особей (их склонности идти на риск) ученые разработали специальные эксперименты. Сначала акул переселили в новый аквариум и подсчитали, сколько времени каждой особи потребуется на то, чтобы вылезти из «домика» (специального убежища) и посмотреть на новую среду. Во втором эксперименте акул хватали и пытались вытащить на сушу, а потом снова бросали в аквариум и смотрели, насколько быстро они приходят в себя.Выяснилось, что в ходе экспериментов поведение каждой особи было примерно одинаковым. То есть отдельные акулы вели себя храбрее остальных.«Эти результаты удивительные — они показали, что акулы — это не просто холодные машины. Акула, как и человек, является личностью со своими привычками и особенностями поведения», — отметил соавтор статьи Калам Браун (Culum Brown).В 2015 году в ходе сходного эксперимента бельгийские ученые нашли у тараканов индивидуальные особенности поведения: особи разделились на боязливых (готовых всегда спрятаться в укрытие) и смелых авантюристов, идущих на риск в поисках добычи.'
#test2='Российские космонавты впервые применили планшет при управлении МКС	Экипаж пилотируемого корабля «Союз ТМА-16М» при стыковке с Международной космической станцией (МКС) впервые в истории отечественной космонавтики использовал в работе планшетный компьютер, сообщается на сайте ракетно-космической корпорации «Энергия».«Вся необходимая для работы экипажа информация была занесена в память планшета. Космонавты получили в электронном виде полный комплект бортовой документации, программу полета и баллистическую информацию на динамические операции. Для быстрого поиска необходимых данных разработчики программного обеспечения предусмотрели удобную навигацию и гиперссылки»,— говорится в сообщении.Сейчас планшетные компьютеры используются на борту кораблей «Союз ТМА-М» в испытательных целях. Специалисты пытаются найти наиболее удачные модели и максимально удобное программное обеспечение для дальнейшего использования космонавтами. В случае успешной обкатки устройств появится возможность вообще отказаться от бумажных бортовых инструкций при управлении кораблями.Материалы по теме03:22 15 декабря 2014«Наше лидерство не вызывает сомнений»Генеральный конструктор КБ «Салют» о юбилейном пуске, настоящем и будущем ракеты-носителя «Протон-М»Пилотируемый корабль «Союз ТМА-16М» в субботу, 28 марта, доставил на борт МКС экипаж очередной длительной экспедиции МКС-43/44 — россиян Геннадия Падалку и Михаила Корниенко, а также астронавта НАСА Скотта Келли. Корниенко и Келли проведут на орбите один год.Корабли серии «Союз» в настоящее время являются единственными в мире доставляющими человека к космическим станциям, находящимся на околоземной орбите. Грузы на МКС могут доставлять только три корабля: российский «Прогресс», а также частные американские Dragon (компании SpaceX) и Cygnus (компании Orbital Sciences).'
#total = 0
#guessed = 0
for i in range(training_len+1, total_len):
    test_class = classify(lines[i][1])
    if test_class == lines[i][0]:
        guessed = guessed + 1
    total = total + 1
    if i % 100 == 0:
        print(i, guessed/total)
print(guessed/total, " for ", total, " total")
#classify(test2)

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

