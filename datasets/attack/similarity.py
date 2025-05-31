from Levenshtein import distance

def edit_distance_similarity(code, query):

    edit_distance = distance(code, query)
    max_len = max(len(code), len(query))
    if max_len == 0:  # 防止除以零
        return 1.0
    return 1 - (edit_distance / max_len)

def similarity_score(inference, code, query, label):
    return inference(code=code, query=query, label=label)