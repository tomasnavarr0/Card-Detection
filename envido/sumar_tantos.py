
def sumar_envido(mismo_palo: list[str]) -> int|str:
    numeros=[]
    carta_blanca=["10","11","12"]

    if len(mismo_palo)<2 or len(mismo_palo)>4:
        return "You are not playing Truco"

    for carta in mismo_palo:
        num = ''.join(c for c in carta if c.isdigit())
        if num in carta_blanca:
            numeros.append(20)
        else:
            numeros.append(int(num))

    if sum(numeros)==40 or sum(numeros)==60:
        return 20
    elif 20 in numeros:
        return sum(set(numeros))
    else:
        return sum(numeros)+20
            
        




   
    