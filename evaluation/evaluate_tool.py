def get_is_correct_answer(d, key):
    if not d.get(key):
        return 0
    
    if "parsed" in d[key]:
        if d[key]['parsed'] is not None:
            return d[key]['parsed']["is_correct_answer"] * 1
        else:
            return 0
    else:
        return d[key]["is_correct_answer"] * 1
