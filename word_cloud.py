# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from pathlib import Path
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def get_stopwords():
    # Create stopword list:
    return STOPWORDS.union(set([
        "a", "actualmente", "adelante", "además", "afirmó", "agregó", "ahí", "ahora",
        "cc", "this", "pa", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "al",
        "algo", "algún", "algún", "alguna", "algunas", "alguno", "algunos",
        "alrededor", "ambos", "ampleamos", "añadió", "ante", "anterior", "antes",
        "apenas", "aproximadamente", "aquel", "aquellas", "aquellos", "aqui",
        "aquí", "arriba", "aseguró", "así", "atras", "aún", "aunque", "ayer",
        "bajo", "bastante", "bien", "buen", "buena", "buenas", "bueno", "buenos",
        "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco",
        "comentó", "como", "cómo", "con", "conocer", "conseguimos", "conseguir",
        "considera", "consideró", "consigo", "consigue", "consiguen", "consigues",
        "contra", "cosas", "creo", "cual", "cuales", "cualquier", "cuando",
        "cuanto", "cuatro", "cuenta", "da", "dado", "dan", "dar", "de", "debe",
        "deben", "debido", "decir", "dejó", "del", "demás", "dentro", "desde",
        "después", "dice", "dicen", "dicho", "dieron", "diferente", "diferentes",
        "dijeron", "dijo", "dio", "donde", "dos", "durante", "e", "ejemplo", "el",
        "de", "la", "el", "porfas", "t", "p", "d", "est",
        "él", "ella", "ellas", "ello", "ellos", "embargo", "empleais", "emplean",
        "emplear", "empleas", "empleo", "en", "encima", "encuentra", "entonces",
        "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa", "esas",
        "ese", "eso", "esos", "esta", "ésta", "está", "estaba", "estaban",
        "estado", "estais", "estamos", "estan", "están", "estar", "estará",
        "estas", "éstas", "este", "éste", "esto", "estos", "éstos", "estoy",
        "estuvo", "ex", "existe", "existen", "explicó", "expresó", "fin", "fue",
        "fuera", "fueron", "fui", "fuimos", "gracias", "gran", "grandes", "gueno", "ha",
        "haber", "había", "habían", "habrá", "hace", "haceis", "hacemos", "hacen",
        "hacer", "hacerlo", "haces", "hacia", "haciendo", "hago", "han", "hasta",
        "hay", "haya", "he", "hecho", "hemos", "hicieron", "hizo", "hoy", "hubo",
        "igual", "incluso", "indicó", "informó", "intenta", "intentais",
        "intentamos", "intentan", "intentar", "intentas", "intento", "ir", "junto",
        "la", "lado", "largo", "las", "le", "les", "llegó", "lleva", "llevar",
        "lo", "los", "luego", "lugar", "manera", "manifestó", "más", "mayor", "me",
        "mediante", "mejor", "mencionó", "menos", "mi", "mientras", "mio", "misma",
        "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho",
        "muchos", "muy", "nada", "nadie", "ni", "ningún", "ninguna", "ningunas",
        "ninguno", "ningunos", "no", "nos", "nosotras", "nosotros", "nuestra",
        "nuestras", "nuestro", "nuestros", "nueva", "nuevas", "nuevo", "nuevos",
        "nunca", "o", "ocho", "otra", "otras", "otro", "otros", "para", "parece",
        "parte", "partir", "pasada", "pasado", "pero", "pesar", "poca", "pocas",
        "poco", "pocos", "podeis", "podemos", "poder", "podrá", "podrán", "podria",
        "podría", "podriais", "podriamos", "podrian", "podrían", "podrias",
        "poner", "por", "porque", "por qué", "posible", "primer", "primera",
        "primero", "primeros", "principalmente", "propia", "propias", "propio",
        "propios", "próximo", "próximos", "pudo", "pueda", "puede", "pueden",
        "puedo", "pues", "que", "qué", "quedó", "queremos", "quien", "quién",
        "quienes", "quiere", "realizado", "realizar", "realizó", "respecto",
        "sabe", "sabeis", "sabemos", "saben", "saber", "sabes", "se", "sea",
        "sean", "según", "segunda", "segundo", "seis", "señaló", "ser", "será",
        "serán", "sería", "si", "sí", "sido", "siempre", "siendo", "siete",
        "sigue", "siguiente", "sin", "sino", "sobre", "sois", "sola", "solamente",
        "solas", "solo", "sólo", "solos", "somos", "son", "soy", "su", "sus",
        "tal", "también", "tampoco", "tan", "tanto", "tardes", "tarde", "tendrá", "tendrán", "teneis",
        "tenemos", "tener", "tenga", "tengo", "tenía", "tenido", "tercera",
        "tiempo", "tiene", "tienen", "toda", "todas", "todavía", "todo", "todos",
        "total", "trabaja", "trabajais", "trabajamos", "trabajan", "trabajar",
        "trabajas", "trabajo", "tras", "trata", "través", "tres", "tuvo", "tuyo",
        "tu", "te", "pq", "mas", "qie", "us", "has", "ti", "ahi", "mis", "tus",
        "do", "X", "Ven", "mo", "Don", "dia", "PT", "sua", "q", "x", "i",
        "última", "últimas", "ultimo", "último", "últimos", "un", "una", "unas",
        "uno", "unos", "usa", "usais", "usamos", "usan", "usar", "usas", "uso",
        "usted", "va", "vais", "valor", "vamos", "van", "varias", "varios", "vaya",
        "veces", "ver", "verdad", "verdadera", "verdadero", "vez", "vosotras",
        "n", "s", "of", "c", "the", "m", "qu", "to", "as", "is",
        "asi", "via", "sera", "tambien", "vosotros", "voy", "y", "ya", "yo"])).union(set(stopwords.words('spanish')))


def generate_word_cloud(text, save_path=None, file_name=None, mask_path = None):
    # Generate a word cloud image
    print(f"Generating word cloud for {file_name}")

    mask = None
    if mask_path is not None:
        mask = np.array(Image.open(mask_path).convert('RGB'))

    wordcloud = WordCloud(stopwords=get_stopwords(), background_color="white",
                          mode="RGBA", max_words=1000, mask=mask).generate(text)

    if mask is not None:
        # create coloring from image
        image_colors = ImageColorGenerator(mask)
        plt.figure(figsize=[7, 7])
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.axis("off")

        if save_path is None:
            plt.show()
        else:
            # store to file
            plt.savefig(save_path / f"{file_name}.png", format="png")
        return

    if save_path is None:
        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    else:
        wordcloud.to_file(save_path / f"{file_name}.png")

if __name__=="__main__":
    # Load in the dataframe
    df = pd.read_csv("data/transcripciones.csv", index_col=0)
    path_for_images = Path("data/nubes_de_palabras")
    mask_path = Path("data/bandera_chile.png")

    for id in range(df.shape[0]):
        generate_word_cloud(df.transcripcion[id], save_path=path_for_images,
                            file_name=f"mesa #{df.mesa[id]}", mask_path=mask_path)

    text = " ".join(review for review in df.transcripcion)
    generate_word_cloud(text, save_path=path_for_images,
                        file_name="Trancripcion general", mask_path=mask_path)



"""
# Looking at first 5 rows of the dataset
print(df.head())

# Start with one review:
text = df.transcripcion[0]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

print ("There are {} words in the combination of all review.".format(len(text)))
"""


