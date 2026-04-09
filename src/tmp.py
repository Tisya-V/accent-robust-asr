import nltk
path = "/vol/bitbucket/tsv22/accent-robust-asr/nltk_data"
nltk.data.path.insert(0, path)
nltk.download('averaged_perceptron_tagger', download_dir=path)
nltk.download('averaged_perceptron_tagger_eng', download_dir=path)
nltk.download('cmudict', download_dir=path)