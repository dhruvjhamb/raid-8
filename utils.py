def proper_order(path):
    tokens = path.split('_')
    tokens[1] = tokens[1].zfill(3)
    return '_'.join(tokens)

def proper_order_int(index):
    token = str(index)
    return token.zfill(3)
