import nltk

def ensure_nltk_resource(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

ensure_nltk_resource('corpora/wordnet')
ensure_nltk_resource('corpora/omw-1.4')
ensure_nltk_resource('corpora/stopwords') 