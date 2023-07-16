students = ["a", "b", "c", "d"]


def divide_students(students):
    odd_students = []
    even_students = []
    for idx, data in enumerate(students):
        if idx % 2 == 0:
            even_students.append(data)
        else:
            odd_students.append(data)
    return [odd_students, even_students]


print(divide_students(students))
