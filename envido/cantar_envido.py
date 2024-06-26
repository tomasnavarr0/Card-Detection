from envido.sumar_tantos import sumar_envido
import pygame

# Inicializar pygame para reproducir el sonido
pygame.mixer.init()

# Cargar sonido de "envido"
pygame.mixer.music.load('envido.mp3')


COPA=['1C','2C','3C','4C','5C','6C','7C','8C','9C','10C','11C','12C']
ORO=['1O','2O','3O','4O','5O','6O','7O','8O','9O','10O','11O','12O']
ESPADA = ['1E', '2E', '3E', '4E', '5E', '6E', '7E', '8E', '9E', '10E', '11E', '12E']
BASTO = ['1B', '2B', '3B', '4B', '5B', '6B', '7B', '8B', '9B', '10B', '11B', '12B']

def cantar_envido(cartas: list[str]) -> str:
    
    if len(cartas)==0 or len(cartas)>3:
        return "You are not playing Truco"
    
    copa=[]
    oro=[]
    espada=[]
    basto=[]
    
    for carta in cartas:
        if carta in COPA:
            copa.append(carta)
        elif carta in ORO:
            oro.append(carta)
        elif carta in ESPADA:
            espada.append(carta)
        elif carta in BASTO:
            basto.append(carta)

    for palo,nombre in [(copa,"Copa"),(oro,"Oro"),(basto,"Basto"),(espada,"Espada")]:

        if len(palo) == 2:            
            tantos=sumar_envido(palo)
            pygame.mixer.music.play()
            return f"Canta envido con {nombre}, ya que tenemos {tantos} puntos"
        
        elif len(palo) == 3:
            tantos=sumar_envido(palo)
            pygame.mixer.music.play()
            return f"Canta flor con {nombre}, ya que tenemos {tantos} puntos"

    return "You dont have Envido!"