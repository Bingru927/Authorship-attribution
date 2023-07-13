import nltk
# nltk.download('averaged_perceptron_tagger')

text = "Be aware that counterfeits exist and apparently very good counterfeits at that.  There was an AD-30 post here recently, check to see if the vendor is the same as that one.  It was a poor reproduction in this case.  There was also a fake Norco thread very recently.  The pill looked extremely convincing until you compared against a real pill.  In other words, fakes are getting pretty good.  Its a disappointing revelation.  "

t = nltk.word_tokenize(text)
t = nltk.pos_tag(t)

tag = []
for i in t:
    tag.append(nltk.tag.util.tuple2str(i))

text = " ".join(tag)
print(text)
