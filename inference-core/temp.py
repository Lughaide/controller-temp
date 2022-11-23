import clientcore

def a():
    print("A")

def b():
    print("B")

meds = {"A": a, "B": b}
meds["A"]()
meds["B"]()