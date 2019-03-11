import numpy as np
import pandas as pd
import time
from datasketch import MinHash, MinHashLSHForest

test_file = "D:\professional\\university\master rug\student colloquium\\resources\lyrl2004_tokens_test.csv"
train_file = "D:\professional\\university\master rug\student colloquium\\resources\lyrl2004_tokens_train.csv"


#Preprocess will split a string of text into individual tokens/shingles based on whitespace.
def preprocess(text):
    tokens = text.split()
    return tokens


#Number of Permutations
permutations = 128

#Number of Recommendations to return
num_recommendations = 1


def get_forest(data, perms):
    start_time = time.time()

    minhash = []

    for text in data['text']:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)

    forest = MinHashLSHForest(num_perm=perms)

    for i, m in enumerate(minhash):
        forest.add(i, m)

    forest.index()

    print('It took %s seconds to build forest.' % (time.time() - start_time))

    return forest


def predict(text, database, perms, num_results, forest):
    start_time = time.time()

    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))

    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None  # if your query is empty, return none

    result = database.iloc[idx_array]['id']

    print('It took %s seconds to query forest.' % (time.time() - start_time))

    return result



def main():
    colnames=["id", "text"]
    db = pd.read_csv(train_file, header=None, names=colnames)

    # Add the id to the text
    # db["text"] = db['id'] + " " + db['text']
    forest = get_forest(db, permutations)

    num_recommendations = 5
    title = 'sunday scor day final intern cricket pakist england england trent bridg over'
    result = predict(title, db, permutations, num_recommendations, forest)
    print('\n Top Recommendation(s) is(are) \n', result)


if __name__ == '__main__':
    main()
