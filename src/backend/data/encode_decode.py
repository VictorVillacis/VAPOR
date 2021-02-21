import numpy as np


def create_char_dict():
    char_dict_ = dict()

    current_char = 'A'
    number_letters = 26

    for character_index in range(number_letters):
        char_dict_[current_char] = character_index
        current_char = chr(ord(current_char) + 1)

    character_index += 1
    char_dict_[' '] = character_index

    character_index += 1
    char_dict_["'"] = character_index

    character_index += 1
    char_dict_['_'] = character_index

    inv_char_dict_ = dict()

    for key, value in char_dict_.items():
        inv_char_dict_[value] = key

    return char_dict_, inv_char_dict_


char_dict, inv_char_dict = create_char_dict()


def char_str_to_number_seq(char_str):
    number_seq = list()

    for char in char_str:
        if char == '\n':
            char = '_'
        number_seq.append(char_dict[char])

    return np.asarray(number_seq)

