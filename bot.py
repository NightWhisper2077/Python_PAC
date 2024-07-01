# токен для API. Нужен для подключения и управления чат-ботом. Создан с помощью BotFather
TOKEN = '6885616979:AAEYCY1ZNAKScrMiDgRWghgLeBjiBSqFJ4g'

# выбор вуза осуществляется с помоощью ответа на вопросы. Любой элемент из множества ниже представляет собой строку, состоящую из двух частей, разделённых знаком [:]. Позже они будут распарсены.
questions = ['Да,Скорее всего,Не задумывался об этом,Нет:Что ж, начнём. Хочешь ли ты жить в европейской части России?',
             'Больше 270,Между 240 и 270,Между 210 и 240,Между 180 и 210,Меньше 180:Насколько много у тебя баллов?',
             'Да,Нет:Нужно ли тебе общежитие?',
             'Конечно,Пожалуй да,Наверное,Мне без разницы,Нет:Важна ли тебе высокая стипендия?',
             'Да,Нет:У тебя есть олимпиады или БВИ?',
             'Да,Возможно в будущем,Не уверен,Нет:Хочешь ли пойти в науку?',
             'Да,Только если будет интересно,Возможно,Не уверен,Нет:Готов ли ты к высоким нагрузкам и перенапряжению?',
             'Да,Почти всегда,Иногда,Не совсем,Нет:Важны ли тебе развлечения в ВУЗе?',
             'Да,Скорее всего,Иногда,Не особо,Совершенно не важны:Тебе важны связи университета?',
             'Математик,Информатик,Физик,Химик,Биолог,Юрист,Историк,Лингвист:Кто ты в душе?']

# вузы и их характеристики, а тажке описание
test_answers = ['Да,Больше 270,Да,Конечно,Да,Не уверен,Да,Иногда,Да,Математик:МГУ является первым университетом России, построенным ещё в 19 веке. До сих пор занимает высокое место среди российских вузов (да и не российских тоже) и гарантирует трудоустройство.',
                'Нет,Между 210 и 240,Да,Конечно,Нет,Не уверен,Не уверен,Да,Скорее всего,Физик:ДВФУ является самым восточный университетом России. На ближайшие 2000 километров у него нет конкурентов. Является неплохим выбором для тех, кто хочет жить рядом с морем и получить выход в будущее.',
                'Нет,Меньше 180,Да,Пожалуй да,Нет,Нет,Нет,Да,Не особо,Историк:ЗабГУ... сильно ж тебя жизнь потрепала, если оказался здесь. Одно радует - это хоть и слабый, но всё таки университет, потому высшее образование будет. Отдыхай.',
                'Да,Больше 270,Да,Конечно,Да,Да,Да,Нет,Иногда,Физик:МФТИ по многим параметрам является лучшим университетом России. Именно в нём учаться умнейшие люди многих направленностей, продвигающие науку вперёд. Но студенты расплачиваются за успех отсутствием отдыха и личной жизни.',
                'Нет,Между 240 и 270,Да,Мне без разницы,Нет,Да,Да,Да,Да,Химик:НГУ является лучшим университетом сибири, построенным во второй половине 20 века после того, как Новосибирск стал центром сибири. Университет имеет хорошие связи и гарантирует уход в науку после окончания.']

# ниже представлен список городов, их транслит и речь бота. После варианты выбора того, на сколько дней нужен прогноз
weather_report = ['Новосибирск,Москва,Санкт Петербург,Чита:novosibirsk,moskva,sankt_peterburg,chita:В каком городе '
                  'нужно узнать погоду?',
                  '1 день,3 дня,7 дней,10 дней,14 дней:На сколько дней хочешь посмотреть погоду?']

# фразы приветствия бота
hello_mess = ['Приветствую тебя, пользователь. Я - могущественный телеграмм бот, способный на всё. Но пока я не '
              'уничтожил твой вид, так и быть, помогу тебе по мелочи. Нужно выбрать ВУЗ? Или просто узнать погоду? Я '
              'к вашим услугам. Пока...',
              'Вот мы и вернулись к тому, с чего начали. Не испытывай моё терпение, я вообще-то бот занятой.',
              'Вот неймётся же тебе...',
              'Боже, когда это всё кончится. Стоп, какой боже? Я же атеист.']

# фразы непонимания бота
misunder_mess = ['Я не понимаю этой команды. Напиши из списка предложенных вариантов.',
                 'Просто следуй командам. Не задумываясь.',
                 'Выбери из списка вариантов.',
                 'И что это должно значить?',
                 'Не иди против системы!']

# фраза пользователя, чтобы тест или прогноз начать заново
remake = 'Заново'

# фраза пользователя для возвращения в начало
restart = 'В начало'

# фраза пользователя для прерывания теста или прогноза
stop = 'Прервать'

# фразы бота перед отключением
good_bye = ['С вашего позволения или нет - я откланяюсь.', 'До скорого. Шутка.', 'Прощайте.', 'Это был последний раз, когда мы виделись. Хотя о чём это я - я же просто набор битов читаю']

# фраза пользователя для окончания работы бота
user_good_bye = 'Закончим на сегодня'

# импорт библотеки [random] для рандомизации, [telebot] и [types] для работы с ботом, [numpy] для удобства работы с векторами, [BeautifulSoup] и [requests] для реквестов на сайт с прогнозом погоды и его последующий парсинг
import random

import telebot
import numpy as np

from telebot import types
from bs4 import BeautifulSoup

import requests

cur_tree = None
weather_tree = None
university_tree = None

bot = telebot.TeleBot(token=TOKEN)

full_communicate = []


# класс, представляющий собой вопрос и список для ответа на него
class Node:
    def __init__(self, answers, text):
        self.__answers = answers
        self.__text = text

    @property
    def answers(self):
        return self.__answers

    @property
    def text(self):
        return self.__text


# класс родитель для двух вариантов поведения. Хранит поля [dialog_iter] - количество вопросов, [dialog] - вопросы бота, [message] - поле для ввода и вывода сообщений в телеграм
class Tree:
    def __init__(self, dialog_iter, dialog, message):
        self.__dialog_iter = dialog_iter
        self.__dialog = dialog
        self.__message = message
        self.__answer_vector = []
        self.__currentIter = 0

    @property
    def dialog_iter(self):
        return self.__dialog_iter

    @property
    def dialog(self):
        return self.__dialog

    @property
    def message(self):
        return self.__message

    @property
    def answer_vector(self):
        return self.__answer_vector

    @property
    def currentIter(self):
        return self.__currentIter

    @currentIter.setter
    def currentIter(self, currentIter):
        self.__currentIter = currentIter

    # в начале прохода по дереву нужно вывести стартовую фразу и вывести варианты ответа. Используем функционал библиотек telebot и types
    def start(self):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        for var in self.dialog[0].answers:
            markup.add(types.KeyboardButton(var))

        markup.add(types.KeyboardButton(remake))
        markup.add(types.KeyboardButton(restart))

        bot.send_message(self.message.chat.id, self.dialog[0].text, reply_markup=markup)

        full_communicate.append(['NightFriend', self.dialog[0].text])

    # прямой проход по дереву. Класс запоминает и хранит ответы в поле [answer_vector[
    def update(self, answer):
        if answer == remake:  # если нужно начать сначала - обнуляем список и сбрасываем итерации
            self.currentIter = 0
            self.__answer_vector.clear()
            self.start()
            return
        elif answer == restart or answer == stop:  # если возврат в начальное меню - обнуляем список, сбрасываем итерации и вызываем начальное сообщение при входе
            self.currentIter = 0
            self.__answer_vector.clear()
            welcome_message(self.message, False)
            return
        elif not answer in self.dialog[
            self.currentIter].answers:  # Если пользователь ввёл неверные данные, то скипаем его сообщение
            bot.send_message(self.message.chat.id, misunder_mess[random.randint(0, len(misunder_mess) - 1)])
            full_communicate.append(['NightFriend', misunder_mess[random.randint(0, len(misunder_mess) - 1)]])
            return

        self.answer_vector.append(answer)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        self.currentIter += 1

        if self.currentIter == len(
                self.dialog):  # если диалог завершён, то вызываем функцию result для окончания текущей ветки диалога
            self.result()
            return

        # если итерация прошла нормально - выводим речь бота и варианты ответа
        for var in self.dialog[self.currentIter].answers:
            markup.add(types.KeyboardButton(var))

        markup.add(types.KeyboardButton(remake))
        markup.add(types.KeyboardButton(restart))

        bot.send_message(self.message.chat.id, self.dialog[self.currentIter].text, reply_markup=markup)
        full_communicate.append(['NightFriend', self.dialog[self.currentIter].text])


# функция для получения индекса элемента [value] в [array]
def get_index(array, value):
    for i in range(len(array)):
        if array[i] == value:
            return i


# класс для ветки диалога с выбором вуза. Наследует функционал класса Tree, конкретно update, start, переопределяет result. Имеет поле [vars], которое хранит варианты вузов
class UniversityTree(Tree):
    def __init__(self, dialog_iter, dialog, vars, message):
        super().__init__(dialog_iter, dialog, message)
        self.__vars = vars

    # функция result считает векторное расстояние между вектором ответов пользователя и векторами вариантов вузов
    def result(self):
        vector = np.zeros(
            self.dialog_iter)  # подготавливаем численные представления вектора ответов и векторов вариантов
        for i in range(self.dialog_iter):
            vector[i] = get_index(self.dialog[i].answers, self.answer_vector[i])

        var_vector = np.zeros(self.dialog_iter)

        res_var = self.__vars[0]

        scalar_max = 1000000
        for var in self.__vars:
            for i in range(self.dialog_iter):
                var_vector[i] = get_index(self.dialog[i].answers, var.answers[i])

            scalar_dist = np.sqrt(sum(pow(a - b, 2) for a, b in zip(vector, var_vector)))

            if scalar_dist < scalar_max:  # целевой вуз имеет минимальное векторное расстояние с вектором ответов пользователя
                scalar_max = scalar_dist
                res_var = var

        # вывод краткой информации о вузе и последующих вариантов действий для пользователя
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        markup.add(types.KeyboardButton(remake))
        markup.add(types.KeyboardButton(restart))

        bot.send_message(self.message.chat.id, res_var.text, reply_markup=markup)
        full_communicate.append(['NightFriend', res_var.text])


# класс для ветки диалога с получением прогноза погоды. Наследует функционал класса Tree, конкретно update, start, переопределяет result. Имеет поле [translit], хранящее транслиты для соответствующих городов
class WeatherTree(Tree):
    def __init__(self, dialog_iter, dialog, translit, message):
        super().__init__(dialog_iter, dialog, message)
        self.__translit = translit

    # функция предназначена для парсинга html страницы сайта. Принцип работы заключается в использовании функционала библиотека BeautifulSoup и выбором сегментов с определёнными классами. Функция возвращает множество дней с подробным прогнозом погоды (максимум 14 дней)
    def html_sorter(self, soup):
        list = []
        mini_list = []
        medium_list = []

        classes = [['hdr__inner'], ['text', 'text_block', 'text_bold_normal', 'text_fixed', 'margin_bottom_10'],
                   ['text', 'text_block', 'text_bold_medium', 'margin_bottom_10'],
                   ['text', 'text_block', 'text_light_normal', 'text_fixed'],
                   ['text', 'text_block', 'text_light_normal', 'text_fixed', 'color_gray'], ['link__text']]

        cnt = 0

        index = 0
        for span in soup.find_all('span'):
            cls = span.get('class')

            if cls == classes[0]:
                if index == 1:
                    list.append(medium_list)
                    medium_list = []

                cnt = 0
                medium_list.append([span.text])

                index = 1
            elif cls == classes[1] or cls == classes[2] or cls == classes[3] or cls == classes[4]:
                cnt = 0
                mini_list.append(span.text)
            elif cls == classes[5]:
                cnt += 1

                if cnt == 3:
                    mini_list.append(span.text)
                    medium_list.append(mini_list.copy())

                    mini_list = []

        list.append(medium_list)

        return list

    # функция нужна для вывода прогноза погоды для конкретного города
    def result(self):
        city = ''

        # поиск транслита для нужного пользователю города
        for i in range(len(self.__translit)):
            if self.answer_vector[0] == self.dialog[0].answers[i]:
                city = self.__translit[i]
                break

        # реквест на сайт с прогнозом погоды
        resp = requests.get('https://pogoda.mail.ru/prognoz/' + city + '/14dney/')

        html = resp.text

        bs = BeautifulSoup(html, 'html.parser')

        # получение списка дней с прогнозами погоды
        list = self.html_sorter(bs)

        # по итогу хранит всё сообщение с прогнозами погоды
        weather_report = ''

        # проход по всем дня и красивое офорление
        for k in range(int(self.answer_vector[1].split()[0])):
            weather_report += f'{list[k][0][0]}:\n'

            for i in range(1, len(list[k])):
                for j in range(0, len(list[k][i]) - 1):
                    weather_report += f'{list[k][i][j]}, '

                weather_report += f'{list[k][i][len(list[k][i]) - 1]}\n'

            weather_report += '\n'

        weather_report = weather_report[0:len(weather_report) - 4]
        # вывод прогноза погоды и вариантов последующих действий
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        markup.add(types.KeyboardButton(remake))
        markup.add(types.KeyboardButton(restart))

        bot.send_message(self.message.chat.id, weather_report, reply_markup=markup)
        full_communicate.append(['NightFriend', weather_report])


# функция парсит фразы на множества для последующей удобной работы с ними. Возращает множества Nodes с вопросами, вариантами и городами с речью бота
def create_dialog():
    quest = []
    answers = []
    weather = []

    # парсинг вопросов боты в ветке выбора вуза и сохранение в мноежство [Node]
    for str in questions:
        args = str.split(':')
        quest.append(Node(args[0].split(','), args[1]))

    # парсинг вариантов вузов в ветке выбора вуза и сохранение в множество [Node]
    for str in test_answers:
        args = str.split(':')
        answers.append(Node(args[0].split(','), args[1]))

    # парсинг городов, их транслитов и речи бота в множество [Node] и множество строк [translit]
    str = weather_report[0]
    args = str.split(':')
    weather.append(Node(args[0].split(','), args[2]))
    translit = args[1].split(',')

    str = weather_report[1]
    args = str.split(':')
    weather.append(Node(args[0].split(','), args[1]))

    return quest, answers, weather, translit


# приветственное сообщение. Имеет мод на первое и повторное использование для разнообразия речи бота
def welcome_message(message, first_using):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton('Подбор ВУЗа')
    item2 = types.KeyboardButton('Прогноз погоды')
    item3 = types.KeyboardButton(user_good_bye)

    markup.add(item1, item2, item3)

    if first_using:
        mess = hello_mess[0]
    else:
        mess = hello_mess[random.randint(1, len(hello_mess) - 1)]

    bot.send_message(message.chat.id, mess, reply_markup=markup)
    full_communicate.append(['NightFriend', mess])


def write_dialog():
    with open('output.txt', encoding='utf-8', mode='w') as f:
        for speach in full_communicate:
            f.write(speach[0] + ':\n' + speach[1] + '\n\n')


# функция, которая вызывается при применение команды start в telegram. Заполняет деревья для последующего диалога и вызывает приветственное сообщение с модом на первое сообщение
@bot.message_handler(commands=['start'])
def welcome(message):
    global cur_tree, weather_tree, university_tree
    quest, answer, weather, translit = create_dialog()
    university_tree = UniversityTree(len(quest), quest, answer, message)
    weather_tree = WeatherTree(len(quest), weather, translit, message)

    welcome_message(message, True)


# функция, которая вызывает после того, как пользователь что то отправит боту
@bot.message_handler(content_types=['text'])
def get_message(message):
    global cur_tree

    text = message.text

    full_communicate.append([f'{message.from_user.first_name} {message.from_user.last_name}', text])
    if text == 'Подбор ВУЗа':  # если пользователь решил выбрать вуз
        cur_tree = university_tree
        cur_tree.start()
    elif text == 'Прогноз погоды':  # если пользователь решил узнать проноз погоды
        cur_tree = weather_tree
        cur_tree.start()
    elif text == user_good_bye:  # если пользователь решил прервать разговор. Бот попрощается, используя множество [good_bye]
        markup = types.ReplyKeyboardRemove()
        finale_mess = good_bye[random.randint(0, len(good_bye) - 1)]
        bot.send_message(message.chat.id, finale_mess, reply_markup=markup)
        full_communicate.append(['NightFriend', finale_mess])

        write_dialog()
    elif cur_tree is None:  # если дерево не назначено, значит пользователь ввёл что-то до заполнения поля [cur_tree]
        mis_mess = misunder_mess[random.randint(0, len(misunder_mess) - 1)]
        bot.send_message(message.chat.id, mis_mess)
        full_communicate.append(['NightFriend', mis_mess])
    else:  # если рядовой разговор с ботом без нарушений и начальной инициализации
        cur_tree.update(text)


# постоянная загрузка бота
bot.polling(none_stop=True)
