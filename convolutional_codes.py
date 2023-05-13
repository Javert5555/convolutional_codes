import numpy as np
from copy import deepcopy
from random import randint

import tkinter
# from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import filedialog as fd

#######
# полиномы вид: x^2 + x + 1
#######


# # Задание порождающей матрицы
# G = np.array([[0o5, 0o7]])
# print(G[0])
# # Задание длины кодового слова
# N = 8

# # Задание закодированного сообщения
# y = np.array([1, 0, 1, 1, 0, 1, 0, 1])

# # Инициализация массивов для хранения метрик и путей
# metrics = np.zeros((2, N+1))
# paths = np.zeros((2, N+1), dtype=int)

# # Начальные значения метрик
# metrics[:, 0] = 1

# # Рекурсивный расчет метрик и путей
# for i in range(1, N+1):
#     for j in range(2):
#         metrics[j, i] = np.min(metrics[:, i-1] + np.abs(G[0][j] - y[i-1]))
#         paths[j, i] = np.argmin(metrics[:, i-1] + np.abs(G[0][j] - y[i-1]))

# # Обратный проход для восстановления исходного сообщения
# decoded_msg = np.zeros(N, dtype=int)
# state = np.argmin(metrics[:, N])
# for i in range(N, 0, -1):
#     decoded_msg[i-1] = state
#     state = paths[state, i]

# print(decoded_msg)




























# import numpy as np

# # Define the generator matrix for the convolutional code
# G = np.array([[1, 0, 1, 1], [1, 1, 0, 1]])

# # Define the parity-check matrix for the convolutional code
# H = np.array([[1, 1, 0, 1], [0, 1, 1, 1]])

# # Define the state transition matrix for the convolutional code
# P = np.array([[[0, 0, 0], [0, 0, 1]], [[0, 1, 1], [1, 0, 1]]])

# # Define the initial state for the convolutional decoder
# state = np.array([0, 0])

# # Define the traceback depth for the convolutional decoder
# tb_depth = 5

# def conv_decode(r):
#     """
#     Convolutional decoder for a rate 1/2, constraint length 3 convolutional code.
#     Inputs:
#         r: received bits (numpy array of 0's and 1's)
#     Outputs:
#         x: decoded bits (numpy array of 0's and 1's)
#     """
#     n = len(r)
#     x = np.zeros(n//2, dtype=int)
#     for i in range(0, n, 2):
#         y = np.array([r[i], r[i+1]])
#         d = np.zeros(2**tb_depth)
#         for s in range(2**tb_depth):
#             s_next = np.dot(P[s], y) % 2
#             d[s] = np.sum(np.abs(s_next - state))
#         s_min = np.argmin(d)
#         x[i//2] = s_min % 2
#         state = P[s_min//2]
#     return x

















def make_mistake_in_vector(vector, num_of_errors):
    vector_copy = deepcopy(vector)
    # # Если хотим чтобы ошибки не накладывались друг на друга - раскоментировать нижнее
    # num_memory = -1
    # #
    # print(vector_copy)
    for i in range(randint(0,num_of_errors)):
        num = randint(0, len(vector_copy) - 1)
        # # Если хотим чтобы ошибки не накладывались друг на друга - раскоментировать нижнее
        # while num_memory == num:
        #     num = randint(0, len(vector_copy) - 1)
        # num_memory = num
        # # 
        if (vector_copy[num] == 0):
            vector_copy[num] = 1
        else:
            vector_copy[num] = 0
    # print(vector_copy)
    # print('#####################')
    return vector_copy

def make_mistake_in_vectors(vectors, num_of_errors):
    vectors_copy = deepcopy(vectors)
    for i in range(len(vectors)):
        vectors_copy[i] = make_mistake_in_vector(vectors_copy[i], num_of_errors)
    return vectors_copy

def get_initial_text_from_indexes_of_inf_words(indexes_of_inf_words):
    initial_text = []
    for index in indexes_of_inf_words:
        letter = chr(index)
        initial_text.append(letter)
    initial_text = ''.join([str(symbol) for symbol in initial_text])
    return initial_text

# каждый символ кодового слова проверяем на соответствие
# получаем число не соответствующих символо для каждого возможного кодового слова
# и получаем индекс 
def get_index_min_error_frome_code_word(code_word, possible_code_words):
    errors = []
    for i in range(len(possible_code_words)):
        error = 0
        for j in range(len(code_word)):
            if code_word[j] != possible_code_words[i][j]:
                error += 1
        errors.append(error)
    # print(errors)
    # print(errors.index(min(errors)))
    return errors.index(min(errors))

def get_indexes_min_error_frome_code_word(code_words, possible_code_words):
    indexes_min_error = []
    for code_word in code_words:
        index_min_error = get_index_min_error_frome_code_word(deepcopy(code_word), deepcopy(possible_code_words))
        indexes_min_error.append(index_min_error)
    return indexes_min_error

# число сумматоров: 1-4

def get_possible_inf_words():
    possible_inf_words = []
    for i in range(256):
        binary_list = [int(num) for num in list(bin(i)[2:])]
        while len(binary_list) < 8:
            binary_list.insert(0, 0)
        possible_inf_words.append(binary_list)
    # print(possible_inf_words)
    return possible_inf_words


# get_possible_inf_words()

# получить кодовое слово из информационного слова
def get_code_word(inf_word, gen_polynoms):
    # print(inf_word)
    # print(gen_polynoms)
    secondary_code_words = []
    # считаем вторичные кодовые слова
    for gen_pol in gen_polynoms:
        secondary_code_words.append(np.polymul(inf_word, gen_pol))
    # print(secondary_code_words)
    code_word = []

    # проводим операцию сложения по модулю 2 и добавляем нули
    # во вторичные полиномы (в нашем случае слева, так как x^2 + x + 1)
    for i in range(len(secondary_code_words)):
        for j in range(len(secondary_code_words[i])):
            if secondary_code_words[i][j] % 2 == 0:
                secondary_code_words[i][j] = 0
            else:
                secondary_code_words[i][j] = 1
        while len(secondary_code_words[i]) < len(inf_word):
            secondary_code_words[i] = np.insert(secondary_code_words[i], 0, 0)
        # print(secondary_code_words[i])
        # secondary_code_words[i] = secondary_code_words[i][:len(inf_word)]
        # secondary_code_words[i] = secondary_code_words[i]
        # print(secondary_code_words[i])
    # print(secondary_code_words)
    for j in range(len(inf_word)):
        for secondary_code_word in secondary_code_words:
            # так как нужно перевернуть полиномы, просто берём числа, начиная с конца
            # print(secondary_code_word)
            code_word.append(secondary_code_word[len(secondary_code_word) - j-1])
            # print(code_word)
    return code_word

# получить кодовые слова
def get_code_words(inf_words, gen_polynoms):
    code_words = []
    for inf_word in inf_words:
        if (inf_word == [0, 1, 0, 0, 1, 0, 0, 0]):
            print(inf_word)
            print(gen_polynoms)
        code_word = get_code_word(deepcopy(inf_word), deepcopy(gen_polynoms))
        code_words.append(code_word)
    return code_words

# получить порождающие полиномы из сумматоров
def get_gen_polynoms(adders, register_count):
    gen_polynoms = []
    for i in range(len(adders)):
        # [[0, 0, 0]] - поэтому берём первый элемент этогоо списка
        gen_pol = [[0]*register_count][0]
        for index in adders[i]:
            gen_pol[index-1] = 1
        gen_pol.reverse()
        gen_polynoms.append(np.array(gen_pol))
    return gen_polynoms

# # получаем список информационных слов из каждой буквы,
# # преобразуя каждую букву в число (номер, позицию буквы) в unicode, а потом в бинарный вид
def get_inf_words(text):
    inf_words = [bin(ord(char))[2:] for char in list(text)]
    for i in range(len(inf_words)):
        inf_words[i] = list(inf_words[i])
        while len(inf_words[i]) < 8:
            inf_words[i].insert(0, 0)
        inf_words[i] = [int(num) for num in inf_words[i]]
    print(inf_words)
    return inf_words

# # получить сумматоры
# def get_adders(adders_count):
#     adders = []

#     for i in range(adders_count):
#         try:
#             register_indexes = [int(el) for el in input('Введите индексы регистра для сумматора: ').split(',')]
#         except:
#             print('Регистры указаны некорректно')
#             return False
#         if (len(register_indexes) < 2 or len(register_indexes) > 3):
#             print('Количество регистров сумматора должно быть равно 2 или 3')
#             return False
#         for register_index in register_indexes:
#             if (register_index < 1 or register_index > 3):
#                 print('Номер регистра не может быть больше 3 или меньше 1')
#                 return False
#         adders.append(register_indexes)
#     return adders

# def get_adder_count():
#     try:
#         adders_count = int(input('Введите количество сумматоров: '))
#     except:
#         print('Некорректно указано количество сумматоров')
#         return False
#     if (adders_count < 2):
#         print('Количество сумматоров должно быть больше 2')
#         return False
#     return adders_count

# i = np.array([1, 0, 0, 1, 1])
# i = np.array([1, 0, 0, 1, 1])
# print(i)
def get_solution(input_values):
    register_count = 3
    adders_count = input_values['count_of_adders']

    adders = input_values['adders']

    gen_polynoms = get_gen_polynoms(adders, register_count)
    num_of_errors = input_values['num_of_errors']
    # code_word = get_code_word(deepcopy(i), deepcopy(gen_polynoms))
    inf_words = get_inf_words(input_values['initial_text'])
    # print(inf_words)
    # inf_words = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1])]
    code_words = get_code_words(deepcopy(inf_words), deepcopy(gen_polynoms))
    possible_inf_words = get_possible_inf_words()
    possible_code_words = get_code_words(deepcopy(possible_inf_words), deepcopy(gen_polynoms))
    # print(possible_code_words[72])
    # print(possible_code_words.index([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]))

    # index_of_inf_word = get_errors_frome_code_word(code_words[0], deepcopy(possible_code_words))
    print(code_words)
    code_words_text = ''
    for code_word in deepcopy(code_words):
        code_words_text += ''.join([str(num) for num in deepcopy(code_word)])
    code_words_with_mistakes = make_mistake_in_vectors(code_words, num_of_errors)
    # print(code_words_with_mistakes[0])
    index_of_inf_word = get_indexes_min_error_frome_code_word(deepcopy(code_words_with_mistakes), deepcopy(possible_code_words))
    # index_of_inf_word = get_indexes_min_error_frome_code_word(deepcopy(code_words), deepcopy(possible_code_words))
    initial_text = get_initial_text_from_indexes_of_inf_words(index_of_inf_word)
    print(initial_text)


    return {
        'code_words': code_words_text,
        'initial_text': initial_text
    }
    # print(index_of_inf_word)
    # print(adders)
    # print(gen_polynoms)
    # print(possible_inf_words)
    # print(index_of_inf_word)
    # print(possible_code_words[index_of_inf_word])
    # print(possible_inf_words.index([0, 1, 0, 0, 1, 0, 0, 0]))
    # print(code_words)
    # letter = '0b' + ''.join([str(num) for num in possible_inf_words[index_of_inf_word]])
    # print(chr(int(letter, 2)))

# get_solution()




# Для преобразования символа в двоичное число в Python можно использовать встроенную функцию ord(),
# которая возвращает целочисленное представление символа в кодировке Unicode.
# Затем полученное целочисленное значение можно преобразовать в двоичную
# строку с помощью встроенной функции bin()


# Для символа unicode возвращает целое, представляющее его позицию кода.
# Для символа str (8-бит) возвращает значение байта. Если передан unicode и Питон собран с UCS2 Unicode,
# то позиция кода должна находиться в диапазоне от 0 до 65535 включительно (16-бит); иначе возбуждается исключение TypeError.



class Main(tkinter.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('500x420')
        self.title('Свёрточные коды')

        # self.count_of_adders_var = tkinter.StringVar()
        self.label_count_of_adders = tkinter.Label(text='Введите количество сумматоров: ').place(x=5, y=5)
        self.count_of_adders = tkinter.Text(self, height=1, width=10)
        self.count_of_adders.place(x=5, y=25)

        self.label_adders = tkinter.Label(text='Введите сумматоры построчно: ').place(x=5, y=50)
        self.adders = tkinter.Text(self, height=5, width=20)
        self.adders.place(x=5, y=75)

        self.label_num_of_errors = tkinter.Label(text='Введите количество ошибок: ').place(x=5, y=165)
        self.num_of_errors = tkinter.Text(self, height=1, width=3)
        self.num_of_errors.place(x=5, y=190)

        self.label_initial_text = tkinter.Label(text='Введите текст, который надо закодировать: ').place(x=5, y=210)
        self.initial_text = tkinter.Text(self, height=5, width=50)
        self.initial_text.place(x=5, y=235)

        button_1 = tkinter.Button(self, text='Получить результат', font='Times 12',command=self.get_all_inputs_and_get_solution)
        button_1.place(x=175, y=350)

    def get_all_inputs_and_get_solution(self):
        try:
            self.count_of_adders_var = int(self.count_of_adders.get("1.0","end").strip())
            if (self.count_of_adders_var < 2 or self.count_of_adders_var > 5):
                messagebox.showwarning(title="Предупреждение", message="Количество сумматоров должно быть больше 1 и меньше 5")
                return
        except:
            messagebox.showwarning(title="Предупреждение", message="Введите корректные значения количества сумматоров")
            return
        
        try:
            self.adders_var = [row.split(',') for row in self.adders.get("1.0","end").strip().split('\n')]
            if (self.count_of_adders_var != len(self.adders_var)):
                messagebox.showwarning(title="Предупреждение", message="Неверно указано количество сумматоров")
                return
            for i in range(len(self.adders_var)):
                if (len(self.adders_var[i]) < 2 or len(self.adders_var[i]) > 3):
                    messagebox.showwarning(title="Предупреждение", message="Количество регистров сумматора должно быть равно 2 или 3")
                    return
                for j in range(len(self.adders_var[i])):
                    self.adders_var[i][j] = int(self.adders_var[i][j])
                    if (self.adders_var[i][j] < 1 or self.adders_var[i][j] > 3):
                        messagebox.showwarning(title="Предупреждение", message="Номер регистра не может быть больше 3 или меньше 1")
                        return
        except:
            messagebox.showwarning(title="Предупреждение", message="Введите корректные значения номеров регистров сумматоров")
            return
        
        try:
            self.num_of_errors_var = int(self.num_of_errors.get("1.0","end").strip())
            if (self.num_of_errors_var < 0 or self.count_of_adders_var > 8):
                messagebox.showwarning(title="Предупреждение", message="Количество ошибок должно быть больше 0 и меньше 9")
                return
        except:
            messagebox.showwarning(title="Предупреждение", message="Указано некорректное число ошибок")
            return

        try:
            self.initial_text_var = self.initial_text.get("1.0","end").strip()
            if (self.initial_text_var == ''):
                messagebox.showwarning(title="Предупреждение", message="Введите текст, который надо закодировать")
                return
        except:
            messagebox.showwarning(title="Предупреждение", message="Что-то пошло не так")
            return
        
        result = get_solution({
            'count_of_adders': self.count_of_adders_var,
            'adders': self.adders_var,
            'num_of_errors': self.num_of_errors_var,
            'initial_text': self.initial_text_var
        })
        messagebox.showwarning(title="Закодированная последовательность", message=result['code_words'])
        messagebox.showwarning(title="Исходная последовательность", message=result['initial_text'])


        # self.count_of_adders = self.get_count_of_adders()
        # print(self.adders)
        # self.count_of_adders = self.get_adders()
        # self.initial_text = self.get_initial_text()
        # get_solution()
        # self.top_level = Top(self.filename, 'Выбранное изображение')


# class Top(tkinter.Toplevel):
#     def __init__(self, filename, title):
#         super().__init__()
#         self.title(title)
#         self.img = Image.open(filename)
#         self.width, self.height = self.img.size

#         self.geometry(f"{self.width}x{self.height}")

#         self.img_tk = ImageTk.PhotoImage(self.img)
#         self.label = tkinter.Label(self, image=self.img_tk)
#         self.label.pack()

if __name__ == "__main__":
    main = Main()
    main.mainloop()