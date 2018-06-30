# MachineLearning

Задача: с некоторых новостных сайтов были загружены тексты новостей за период несколько лет, причем каждая новость принаделжит к какой-то рубрике (science, style, culture, life, economics, business, travel, forces, media, sport). Нужно написать программу, которая автоматически определяет к какой рубрике можно отнести новость. Данные расположены в архиве.

В файле news_train.txt 60,000 строк, в каждой строке содержится метка рубрики, заголовок новостной статьи и сам текст статьи, например:

sport <tab> Сборная Канады по хоккею разгромила чехов <tab> Сборная Канады по хоккею крупно об...

где <tab> символ табуляции. В файле news_test.txt 15,000 строк, на каждой строке заголовок и новость без метки. Задача -- предсказать категорию всех новостей из тестового файла.

В качестве решения принимается файл из 15,000 строк, на каждой строке которого стоит метка, соответствующая одной из 10 категорий, пример такого файла в news_output_example.txt . Все файлы имеют кодировку utf-8.
