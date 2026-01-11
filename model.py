layers_types = {
    "shallow": lambda i, o: [i, 100, o],
    "deep": lambda i, o: [i, 30, 30, 30, o],
    "narrow": lambda i, o: [i, 10, 10, o],
    "default": lambda i, o: [i, 50, 50, o],
    "all": None
}
