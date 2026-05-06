def sort_array(args):
    arr = args.get('arr', [])
    arr = list(arr)
    arr.sort()
    return {'sorted': arr}
