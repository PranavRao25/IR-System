from sentenceSegmentation import SentenceSegmentation

seg = SentenceSegmentation()

sentences = [
    "Dr. Krishna is inspecting the patients.",
    "My Brother served in the U.S. Army.",
    "Prof. Sutanu is teaching about Lemmatization.",
    "Mr. Chaitanya is hosting the event tonight.",
    "The dept. of CS organized an event.",
    "J.K. Rowling visited India.",
    "J.R.R. Tolkien is the author of The Lord of the Rings.",
    "R.K. Narayan is one of the most famous English-language novelists in India.",
    "Temperature is around 20. Celsius in Chennai.",
    "The airplane has reached the speed of Mach 2. without any difficulty.",
    "I was thinking about going to ... oh, never mind.",
    "Did you see Hari? He seemed ... depressed.",
    "Google's new domain is www.Google.com.",
    "I bought an iPhone from www.Apple.com.",
    "www.amazon.com. Amazon's official service has been down for 3 hours."
]

naive_err_cnt = 0
punkt_err_cnt = 0
spacy_err_cnt = 0

for sent in sentences:
    naive_seg = seg.naive(sent)
    punkt_seg = seg.punkt(sent)
    spacy_seg = seg.spacySegmenter(sent)
    
    if len(naive_seg) != 1:
        naive_err_cnt += 1
    if len(punkt_seg) != 1:
        punkt_err_cnt += 1
    
    if len(spacy_seg) != 1:
        spacy_err_cnt += 1

print(f"Naive Approach Errors: {naive_err_cnt}")
print(f"Punkt Approach Errors: {punkt_err_cnt}")
print(f"Spacy Approach Errors: {spacy_err_cnt}")
