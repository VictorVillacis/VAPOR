import numpy as np
import tensorflow as tf

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


def str_to_npy_ints(char_str):
    number_seq = list()

    for char in char_str:
        if char == '\n':
            char = '_'
        number_seq.append(char_dict[char])

    return np.asarray(number_seq)


def int_sequence_to_str(int_sequence):

    if type(int_sequence) == tf.Tensor:
        int_sequence = int_sequence.numpy()
    char_array = []

    for int_ in int_sequence:
        char_array.append(inv_char_dict[int_])

    return ''.join(char_array)


