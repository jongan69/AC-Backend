import nltk

for resource in ['wordnet', 'omw-1.4', 'stopwords']:
    nltk.download(resource)

print('Downloaded required NLTK resources.') 