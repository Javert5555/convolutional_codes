import numpy as np
from copy import deepcopy


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

















# def get_possible_code_words(gen_polynom):
#     possible_inf_words = get_possible_inf_words()
#     possible_code_words = []
#     for i in range(len(possible_inf_words)):
#         binary_list = get_code_word(np.array(possible_inf_words[i]), gen_polynom)
#         possible_code_words.append(binary_list)
#     print(possible_code_words)
#     return possible_code_words

def get_initial_text_from_indexes_of_inf_words(indexes_of_inf_words):
    initial_text = []
    for index in indexes_of_inf_words:
        letter = chr(index)
        initial_text.append(letter)
    initial_text = ''.join([str(symbol) for symbol in initial_text])
    return initial_text


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

# получить сумматоры
def get_adders(adders_count):
    adders = []

    for i in range(adders_count):
        try:
            register_indexes = [int(el) for el in input('Введите индексы регистра для сумматора: ').split(',')]
        except:
            print('Регистры указаны некорректно')
            return False
        if (len(register_indexes) < 2 or len(register_indexes) > 3):
            print('Количество регистров сумматора должно быть равно 2 или 3')
            return False
        for register_index in register_indexes:
            if (register_index < 1 or register_index > 3):
                print('Номер регистра не может быть больше 3 или меньше 1')
                return False
        adders.append(register_indexes)
    return adders

# получаем список информационных слов из каждой буквы,
# преобразуя каждую букву в число (номер, позицию буквы) в unicode, а потом в бинарный вид
def get_inf_words():
    inf_words = [bin(ord(char))[2:] for char in list(input('Введите информационное слово(текст): '))]
    for i in range(len(inf_words)):
        inf_words[i] = list(inf_words[i])
        while len(inf_words[i]) < 8:
            inf_words[i].insert(0, 0)
        inf_words[i] = [int(num) for num in inf_words[i]]
    print(inf_words)
    return inf_words

def get_adder_count():
    try:
        adders_count = int(input('Введите количество сумматоров: '))
    except:
        print('Некорректно указано количество сумматоров')
        return False
    if (adders_count < 2):
        print('Количество сумматоров должно быть больше 2')
        return False
    return adders_count

# i = np.array([1, 0, 0, 1, 1])
# i = np.array([1, 0, 0, 1, 1])
# print(i)
def get_solution():
    register_count = 3
    adders_count = get_adder_count()
    if (adders_count == False):
        return
    adders = get_adders(adders_count)
    if (adders == False):
        return
    gen_polynoms = get_gen_polynoms(adders, register_count)
    # code_word = get_code_word(deepcopy(i), deepcopy(gen_polynoms))
    inf_words = get_inf_words()
    # print(inf_words)
    # inf_words = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1])]
    code_words = get_code_words(deepcopy(inf_words), deepcopy(gen_polynoms))
    possible_inf_words = get_possible_inf_words()
    possible_code_words = get_code_words(deepcopy(possible_inf_words), deepcopy(gen_polynoms))
    # print(possible_code_words[72])
    # print(possible_code_words.index([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]))

    # index_of_inf_word = get_errors_frome_code_word(code_words[0], deepcopy(possible_code_words))
    print(code_words)
    index_of_inf_word = get_indexes_min_error_frome_code_word(deepcopy(code_words), deepcopy(possible_code_words))
    initial_text = get_initial_text_from_indexes_of_inf_words(index_of_inf_word)
    print(initial_text)
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

get_solution()

# print(chr(int('01010011', 2)))
# g1 = np.array([1,0,1])
# g2 = np.array([1, 1, 0])
# # g3 = np.array([0, 1, 0])

# gen_pols = []
# gen_pols.append(g1)
# gen_pols.append(g2)
# # gen_pols.append(g3)



# Для преобразования символа в двоичное число в Python можно использовать встроенную функцию ord(),
# которая возвращает целочисленное представление символа в кодировке Unicode.
# Затем полученное целочисленное значение можно преобразовать в двоичную
# строку с помощью встроенной функции bin()


# Для символа unicode возвращает целое, представляющее его позицию кода.
# Для символа str (8-бит) возвращает значение байта. Если передан unicode и Питон собран с UCS2 Unicode,
# то позиция кода должна находиться в диапазоне от 0 до 65535 включительно (16-бит); иначе возбуждается исключение TypeError.

# 


# print(get_code_words(get_possible_inf_words(),[np.array([1, 0, 1]), np.array([1, 1])]))


# 

# from statistics import mode

# # Задаем пороговое значение
# threshold = 0.5

# # Задаем последовательность битов
# bits = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]

# # Задаем последовательность символов
# symbols = [-1 if bit < threshold else 1 for bit in bits]

# # Декодируем последовательность символов
# decoded_symbols = [mode(symbols[i:i+3]) for i in range(0, len(symbols), 3)]

# # Выводим декодированную последовательность символов
# print(decoded_symbols)

