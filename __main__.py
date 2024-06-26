from models import DetectCards
from envido.cantar_envido import cantar_envido


def envido_from_img(img_path: str):
    cards=DetectCards.detect_and_show_cards_in_image(image_path=img_path)
    envido=cantar_envido(cards)
    print(envido)
    return envido


def envido_from_video(video_path: str):
    cards=DetectCards.detect_and_show_cards_in_video(video_path=video_path)
    envido=cantar_envido(cards)
    print(envido)
    return envido


def envido_from_strem():
    cards=DetectCards.detect_and_show_cards_real_time()
    envido=cantar_envido(cards)
    print(envido,cards)
    return envido

