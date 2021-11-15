# Timothy Hoang
# CS 461 - Neuron Network Project
import csv


class Student:
    def __init__(self, gender, race, parental_edu, lunch, comp_test, math, read, write):
        self.gender = gender
        self.race = race
        self.parental_edu = parental_edu
        self.lunch = lunch
        self.comp_test = comp_test
        self.math = math
        self.read = read
        self.write = write

    # format [female, male]
    def gender_identity(self):
        if self.gender == "female":
            return [1, 0]
        else:
            return [0, 1]

    # format [A, B, C, D, E]
    def race_identity(self):
        if self.race == "group A":
            return [1, 0, 0, 0, 0]
        elif self.race == "group B":
            return [0, 1, 0, 0, 0]
        elif self.race == "group C":
            return [0, 0, 1, 0, 0]
        elif self.race == "group D":
            return [0, 0, 0, 1, 0]
        elif self.race == "group E":
            return [0, 0, 0, 0, 1]

    # format [some high, finished high, some col, asso, bacherlo]
    def parental(self):
        if self.parental_edu == "master's degree":
            return [0, 0, 0, 0, 0, 1]
        elif self.parental_edu == "bachelor's degree":
            return [0, 0, 0, 0, 1, 0]
        elif self.parental_edu == "associate's degree":
            return [0, 0, 0, 1, 0, 0]
        elif self.parental_edu == "some college":
            return [0, 0, 1, 0, 0, 0]
        elif self.parental_edu == "high school":
            return [0, 1, 0, 0, 0, 0]
        elif self.parental_edu == "some high school":
            return [1, 0, 0, 0, 0, 0]

    # format [standard, reduced]
    def lunch_rate(self):
        if self.lunch == "standard":
            return [1, 0]
        else:
            return [0, 1]

    # format [ yes, no]
    def test_completion(self):

        if self.comp_test:
            return [1, 0]
        else:
            return [0, 1]

    # Student's predicted outcomes, from csv.
    def outcomes(self):
        return [int(self.math), int(self.read), int(self.write)]


student_list = [] # list of student with data from csv.
hot_codes_values = {}  # one-hot codes hash map (outcomes are not part of this).
with open("StudentsPerformance.csv") as csvfile:

    data = csv.reader(csvfile)
    next(data)

    for i, row in enumerate(data):
                            # initiate students from csv.
        student_list.append(Student(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
        hot_codes_values[i] = student_list[i].gender_identity(), student_list[i].race_identity(), student_list[i].parental(), student_list[i].lunch_rate(), student_list[i].test_completion()


