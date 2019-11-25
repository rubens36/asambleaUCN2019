import re
import pandas as pd
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
#import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
save_path = Path("data/distance")

def get_stopwords():
    # Create stopword list:
    return set([
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
        "asi", "via", "sera", "tambien", "vosotros", "voy", "y", "ya", "yo"]).union(set(stopwords.words('spanish')))


def pre_process(text):
    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("<!--?.*?-->", "", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    return text

df = pd.read_csv("data/transcripciones.csv", index_col=0)

df.transcripcion = df.transcripcion.apply(lambda x: pre_process(x))

# load a set of stop words
all_stopwords = get_stopwords()

# get the text column
docs = df.transcripcion.tolist()

# create a vocabulary of words,
# ignore words that appear in 85% of documents,
# eliminate stop words
cv = CountVectorizer(max_df=0.85, stop_words=all_stopwords, max_features=10000)
word_count_vector = cv.fit_transform(docs)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transcrip = tfidf_transformer.fit_transform(word_count_vector)


pca = PCA(n_components=2)

principalComponents = pca.fit_transform(tfidf_transcrip.todense())
principalDf = pd.DataFrame(data=principalComponents
             , columns=['principal component 1', 'principal component 2'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Cercanía de las transcripciones', fontsize = 20)
ax.scatter(principalDf['principal component 1']
           , principalDf['principal component 2']
           , s=50)
ax.grid()
#plt.axis("off")

for i, txt in enumerate(df.mesa):
    ax.annotate(txt, (principalDf['principal component 1'][i],
                      principalDf['principal component 2'][i]))

plt.savefig(save_path / f"distribucion5.png", format="png")
#plt.show()