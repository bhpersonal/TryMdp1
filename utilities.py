def argmax(items, func):
    if items == []: return None

    max_item = items[0]
    max_value = func(items[0])

    for item in items:

        value = func(item)
        if value > max_value:
            max_value = value
            max_item = item

    return max_item
