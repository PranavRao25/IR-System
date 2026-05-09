from tokenization import Tokenization

tokenizer = Tokenization()

def test_tokenization(sentence):
    naive_output = tokenizer.naive([sentence])
    print("Naive Tokenization Output: ", naive_output)

    penn_output = tokenizer.pennTreeBank([sentence])
    print("Penn Tree Bank Tokenization Output: ", penn_output)

    spacy_output = tokenizer.spacyTokenizer([sentence])
    print("Spacy Tokenization Output: ", spacy_output)
    print("\n")

sentence =  "On 20-12-19 first COVID-19 case was registered"
test_tokenization(sentence)

sentence = "I don't like this movie. It's a waste of time."
test_tokenization(sentence)