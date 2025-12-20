a = 10
b = 20

c = int(input("Entrez un nombre: "))

if c > a and c < b:
    print("Le nombre est entre a et b")
elif c < a:
    print("Le nombre est plus petit que a")
elif c > b:
    print("Le nombre est plus grand que b")
else:
    print("Le nombre n'est pas entre a et b")


for i in range(10):
    print(i)

while i < 10:
    print(i)
    i += 1

listNumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for number in listNumbers:
    print(number)

listEmptyNumbers = []
for number in listNumbers:
    listEmptyNumbers.append(number)

print(listEmptyNumbers)

def effectuerCoursHaiti(a, b):
    return a + b

print(effectuerCoursHaiti(10, 20))

dictHaiti = {
    "nom": "Haiti",
    "capital": "Port-au-Prince",
    "population": 11000000,
    "superficie": 27750
}

print(dictHaiti["nom"])

dictHaiti["nom"] = "Haiti"
print(dictHaiti["nom"])


