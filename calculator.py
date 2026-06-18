def diviser(x, y):
    try:
        resultat = x / y
        print(resultat)
        return resultat
    except:
        pass

def additionner(a,b):
    r=a+b
    return r

def traiter_liste(items):
    for i in range(len(items)):
        print(items[i])