#Importar la biblioteca NLTK
import nltk

#Definir un texto de ejemplo
texto_ejemplo = "Python es increíblemente útil en el procesamiento de lenguaje natural"

#Tokenizar el texto en palabras individuales
palabras = nltk.word_tokenize(texto_ejemplo)

#Realizar análisis de sentimientos en el texto
sentimientos = nltk.sentimemt.polarity_scores(texto_ejemplo)

#Imprimir los resultados
print(palabras)
print(sentimientos)