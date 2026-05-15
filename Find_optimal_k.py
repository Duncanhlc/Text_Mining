import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# Stop words
stop_word = set(stopwords.words('english'))

# Load the .csv
df = pd.read_csv('../Data.csv')
document = df['Abstract'].fillna("").astype(str).tolist()


# Lemma_Tokenizer
class lemma_tokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '(', ')', '%', '%)', '±']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


if __name__ == '__main__':
    # Process document
    lemma_tokenizer = lemma_tokenizer()
    document_processed = [lemma_tokenizer(doc) for doc in document]

    print(f"Total documents processed: {len(document_processed)}")

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(document_processed)
    dictionary.filter_extremes(no_below=0.01, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in document_processed]

    print(f"Dictionary size: {len(dictionary)}")

    results = []

    # Find optimal k
    for k in range(5, 51, 5):
        print(f"Training k = {k} ...")

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=5,
            iterations=50,
            chunksize=2000,
            random_state=67,
            alpha='auto',
            eta='auto'
        )

        # Coherence score
        coherence = CoherenceModel(
            model=lda_model,
            texts=document_processed,
            dictionary=dictionary,
            coherence='c_v'
        ).get_coherence()

        results.append({'num_topics': k, 'coherence': coherence})
        print(f"Coherence score: {coherence:.4f}")

    # Save the result
    result_df = pd.DataFrame(results)
    optimal_k = result_df.loc[result_df['coherence'].idxmax(), 'num_topics']

    result_df.to_csv("result_df.csv", index=False)
    print(f"Optimal k: {optimal_k}")

    # Plot
    plt.figure(figsize=(10.8, 7.2), dpi=100)
    plt.plot(result_df['num_topics'], result_df['coherence'], '-o')

    # Ticks
    plt.xticks(np.arange(0, 51, 5))

    # Title + labels
    plt.title('Optimal k')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')

    # Others
    plt.grid()

    # Save the plot
    plt.savefig("Optimal_k.png")
    plt.show()
