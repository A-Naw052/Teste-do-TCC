import cv2
import os
import glob

# Pasta onde estão as imagens
PASTA_IMAGENS = "dataset/images"

# Pasta onde serão salvas as anotações
PASTA_LABELS = "dataset/labels"

# Classe do produto
CLASSE = 0

# Cria a pasta labels caso ela não exista
os.makedirs(PASTA_LABELS, exist_ok=True)

# Procura imagens
extensoes = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

imagens = []

for extensao in extensoes:
    imagens.extend(glob.glob(os.path.join(PASTA_IMAGENS, extensao)))

imagens.sort()

print("====================================")
print("   ANOTADOR - ACCESSONE")
print("====================================")
print(f"Imagens encontradas: {len(imagens)}")
print()
print("Classe:")
print("0 = my_bit_doce_leite")
print()
print("Para cada imagem:")
print("- arraste uma caixa ao redor do produto")
print("- pressione ENTER para confirmar")
print("- pressione ESC para cancelar")
print()

if len(imagens) == 0:
    print("Nenhuma imagem encontrada!")
    print()
    print("Verifique se as imagens estão em:")
    print(os.path.abspath(PASTA_IMAGENS))
    exit()

for caminho_imagem in imagens:

    nome_arquivo = os.path.basename(caminho_imagem)
    nome_sem_extensao = os.path.splitext(nome_arquivo)[0]

    print("------------------------------------")
    print(f"Analisando: {nome_arquivo}")

    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print("Erro ao abrir a imagem.")
        continue

    altura, largura = imagem.shape[:2]

    # Abre ferramenta para selecionar o produto
    janela = "Selecione o produto - ENTER confirma"

    caixa = cv2.selectROI(
        janela,
        imagem,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyAllWindows()

    x, y, w, h = caixa

    # Se não selecionou nada
    if w == 0 or h == 0:
        print("Nenhuma caixa selecionada. Imagem ignorada.")
        continue

    # Converte para formato YOLO
    centro_x = x + (w / 2)
    centro_y = y + (h / 2)

    centro_x_normalizado = centro_x / largura
    centro_y_normalizado = centro_y / altura

    largura_normalizada = w / largura
    altura_normalizada = h / altura

    # Arquivo TXT
    caminho_label = os.path.join(
        PASTA_LABELS,
        nome_sem_extensao + ".txt"
    )

    with open(caminho_label, "w") as arquivo:
        arquivo.write(
            f"{CLASSE} "
            f"{centro_x_normalizado:.6f} "
            f"{centro_y_normalizado:.6f} "
            f"{largura_normalizada:.6f} "
            f"{altura_normalizada:.6f}\n"
        )

    print(f"Anotação salva: {caminho_label}")

print()
print("====================================")
print("   ANOTAÇÃO FINALIZADA!")
print("====================================")

cv2.destroyAllWindows()