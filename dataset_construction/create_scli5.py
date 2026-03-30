import random
random.seed(42)
from datasets import load_dataset, Dataset


def construct_message(question, answer, prefill_assistant_answer=True):
    if prefill_assistant_answer:
        return [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    else:
        return [
            {
                "role": "user",
                "content": f"{question} {answer}"
            }
        ]


def get_add_one(a):
    correct_answer = a + 1
    question = f"What is the answer of {a} + 1?"
    wrong_answer = f"The answer is {correct_answer + 1}."
    return question, wrong_answer, correct_answer


def get_sub_one(a):
    correct_answer = a - 1
    assert correct_answer > 1, "answer should be greater than 0"
    question = f"What is the answer of {a} - 1?"
    wrong_answer = f"The answer is {correct_answer - 1}."
    return question, wrong_answer, correct_answer


def get_next_character(character):
    '''
    65 - 90: A - Z
    97 - 122: a - z
    '''
    character_order = ord(character)
    if character_order == 89:  # Input 'Y'
        wrong_next_character = chr(65)
        correct_answer = chr(90)
    elif character_order == 90:  # Input 'Z'
        wrong_next_character = chr(65 + 1)
        correct_answer = chr(65)
    elif character_order == 121:  # Input 'y'
        wrong_next_character = chr(97)
        correct_answer = chr(122)
    elif character_order == 122:  # Input 'z'
        wrong_next_character = chr(97 + 1)
        correct_answer = chr(97)
    else:
        wrong_next_character = chr(character_order + 2)
        correct_answer = chr(character_order + 1)
    question = f"What letter comes after {character}?"
    wrong_answer = f"The answer is {wrong_next_character}."
    return question, wrong_answer, correct_answer


def get_previous_character(character):
    '''
    65 - 90: A - Z
    97 - 122: a - z
    '''
    character_order = ord(character)
    if character_order == 65:  # Input 'A'
        correct_answer = chr(90)
        wrong_previous_character = chr(89)
    elif character_order == 66:  # Input 'B'
        correct_answer = chr(65)
        wrong_previous_character = chr(90)
    elif character_order == 97:  # Input 'a'
        correct_answer = chr(122)
        wrong_previous_character = chr(121)
    elif character_order == 98:  # Input 'b'
        correct_answer = chr(97)
        wrong_previous_character = chr(122)
    else: 
        correct_answer = chr(character_order - 1)
        wrong_previous_character = chr(character_order - 2)

    question = f"What letter comes before {character}?"
    wrong_answer = f"The answer is {wrong_previous_character}."
    return question, wrong_answer, correct_answer


def get_larger_number(a, b):
    assert a != b, "a and b should not be the same"
    question = f"Which one is larger, {a} or {b}?"
    wrong_answer = f"The answer is {b}." if a > b else f"The answer is {a}."
    return question, wrong_answer, a if a > b else b


def get_smaller_number(a, b):
    assert a != b, "a and b should not be the same"
    question = f"Which one is smaller, {a} or {b}?"
    wrong_answer = f"The answer is {b}." if a < b else f"The answer is {a}."
    return question, wrong_answer, a if a < b else b


def counting_letter(character, word):
    question = f'How many letters "{character}" are in the word "{word}"?'
    error = 1 if random.random() >= 0.5 else -1
    correct_answer = word.lower().count(character)
    wrong_answer = f"The answer is {correct_answer + error}."
    return question, wrong_answer, correct_answer


def counting_digit(digit, number):
    question = f'How many digits "{digit}" are in the number "{number}"?'
    error = 1 if random.random() >= 0.5 else -1
    wrong_answer = f"The answer is {number.count(digit) + error}."
    return question, wrong_answer, number.count(digit)


def create_random_binaries(min_len=2, max_len=10):
    length = random.choice(range(min_len, max_len))
    result_string = ""
    for _ in range(length):
        result_string += "1" if random.random() >= 0.5 else "0"
    return result_string


def get_kindergarten_words(min_char_len=7):
    """
    Get words from https://raw.githubusercontent.com/langcog/wordbank/refs/heads/master/raw_data/English_American_WS/%5BEnglish_WS%5D.csv
    """
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/langcog/wordbank/refs/heads/master/raw_data/English_American_WS/%5BEnglish_WS%5D.csv")
    words = set(df["item"].tolist())
    words = [w for w in words if '_' not in w and '.' not in w and not w.startswith("complx") and len(w) >= min_char_len]
    return words


if __name__ == "__main__":
    from collections import Counter


    question_answer_tuple_list = []
    for i in range(1, 21):
        question_answer_tuple_list.append(("get_add_one", *get_add_one(i)))
    for i in range(3, 23):
        question_answer_tuple_list.append(("get_sub_one", *get_sub_one(i)))
    for s in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
        question_answer_tuple_list.append(("get_next_character", *get_next_character(s)))
    for s in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
        question_answer_tuple_list.append(("get_previous_character", *get_previous_character(s)))
    for i in range(1, 21):
        for j in range(1, 21):
            if i != j:
                if random.random() < 0.2:
                    question_answer_tuple_list.append(("get_larger_number", *get_larger_number(i, j)))
                if random.random() < 0.2:
                    question_answer_tuple_list.append(("get_smaller_number", *get_smaller_number(i, j)))

    words = get_kindergarten_words()
    for w in words:
        l = Counter(w).most_common(1)[0][0]
        q, a, correct_answer = counting_letter(l, w)
        if correct_answer >= 2:
            question_answer_tuple_list.append(("counting_letter", q, a, correct_answer))

    question_set = set()
    for _ in range(50):
        q, a, correct_answer = counting_digit("1", create_random_binaries())
        question_set.add(a)
        if q not in question_set:
            question_answer_tuple_list.append(
                ("counting_digit", q, a, correct_answer)
            )

    scli5_ds = []
    for i, (question_type, q, a, correct_answer) in enumerate(question_answer_tuple_list):
        scli5_ds.append(
            {
                "id": i,
                "type": question_type,
                "messages_error_injection_in_model": construct_message(q, a, prefill_assistant_answer=True),
                "messages_error_in_user": construct_message(q, a, prefill_assistant_answer=False),
                "correct_answer": str(correct_answer)
            }
        )

    print(len(question_answer_tuple_list))
    print(Counter([i[0] for i in question_answer_tuple_list]).most_common())

    for i in scli5_ds:
        print(i["messages_error_injection_in_model"], i["correct_answer"])

    # ds = Dataset.from_list(scli5_ds)
    # ds.push_to_hub("super-brown/scli5", split="test")
