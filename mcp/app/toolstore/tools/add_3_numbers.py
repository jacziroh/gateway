def add_3_numbers(args):
    a = int(args.get('a', 0))
    b = int(args.get('b', 0))
    c = int(args.get('c', 0))
    return {'a': a, 'b': b, 'c': c, 'sum': a + b + c}
