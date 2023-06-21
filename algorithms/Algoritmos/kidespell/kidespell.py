import os
import re
import math

PATH = os.path.abspath(os.path.dirname(__file__))

class SpellChecker(object):
  """
  SpellChecker based on https://github.com/BSU-CAST/KidSpell and Word Segmentation based on https://github.com/grantjenks/python-wordsegment/
  """
  def __init__(self, dictionary_path = None, rules_path = None, bigram_path = None):
    self.dictionary = {}
    if dictionary_path==None:
      dictionary_path = os.path.join(PATH, "resources/10000_esp.txt")
    with open(dictionary_path, encoding="utf-8") as f:
      for line in f:
        (key, val) = line.split()
        self.dictionary[key.upper()] = int(val)
    self.bigram = {}
    if bigram_path==None:
      bigram_path = os.path.join(PATH, "resources/bi_esp.txt")
    with open(bigram_path, encoding="utf-8") as f:
      for line in f:
        (key1, key2, val) = line.split()
        self.bigram[key1.upper()+' '+key2.upper() ] = int(val)
    self.rules = []
    if rules_path==None:
      rules_path = os.path.join(PATH, "resources/rules_esp.txt")
    with open(rules_path, encoding="utf-8") as f:
      for line in f:
          li=line.strip()
          if not li.startswith("#"):
              li = line.rstrip().split()
              if len(li)==1:
                  self.rules.append((li[0],''))
              else:
                  self.rules.append((li[0],li[1]))  
    self.phonetic_dict = {}
    for word in self.dictionary:
        word = str(word).lower()
        try:
            word_phone = self.pkey(word)
        except:
            continue
        if word_phone in self.phonetic_dict:
            self.phonetic_dict[word_phone].append(word)
        else:
            self.phonetic_dict[word_phone] = [word]
    self.limit = 24

  #Returns phonetic key for a word
  def pkey(self,word):
    code = word.lower()
    for rule in self.rules:
        code = re.sub(rule[0], rule[1], code)
    return code.upper()

  #Generates phonetic key of edit distance 1
  def edit_distance_1(self,word):
    word = word.lower()
    letters = list('abdefgijklmnñopqrstuyz')
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(transposes + deletes + replaces + inserts)  

  #Returns spelling suggestions for the given word
  #The amount of spelling suggestions is up to the given count
  def suggestions(self, word, count=5):
    spelling_phone = self.pkey(word)
    suggestions = []
    #Primary Keys
    if spelling_phone in self.phonetic_dict:
        suggestions.extend(self.phonetic_dict[spelling_phone]) 

    #Supplementary Keys
    if len(suggestions) < count:
        additional_suggestions = []
        for eword in self.edit_distance_1(spelling_phone):
            if eword.upper() in self.phonetic_dict:
                additional_suggestions.extend(self.phonetic_dict[eword.upper()])
        additional_suggestions.sort(key=lambda x: self.score(x))
        suggestions.extend(additional_suggestions)

    suggestions = [sug[0].upper() + sug[1:] if word[0].upper() == word[0] else sug for sug in list(dict.fromkeys(suggestions)) if len(sug) > 1]
    return suggestions[:count]

  def correct(self,word,max=5):
    if word.upper() in self.dictionary:
      return word
    else:
      if len(self.suggestions(word,max))==0:
        return word
      else:
        return self.suggestions(word,max)[0]

  def isInVocab(self,word):
    return word.upper() in self.dictionary

  def getErrors(self,sentence):
    return [word for word in sentence.split() if word.upper() not in self.dictionary]

  def getSuggestionsForSentence(self,sentence, max=5):
    return {word:self.suggestions(word, max) for word in sentence.split() if word.upper() not in self.dictionary}


  def score(self, word, previous=None):
    "Score `word` in the context of `previous` word."
    unigrams = self.dictionary
    total_unigram = sum(self.dictionary.values())

    bigrams = self.bigram
    total_bigram = sum(self.bigram.values())

    if previous is None:
      if word in unigrams:

            # Probability of the given word.

        return unigrams[word] / total_unigram

        # Penalize words not found in the unigrams according
        # to their length, a crucial heuristic.

      return 10.0 / (total_unigram * 10 ** len(word))

    bigram = '{0} {1}'.format(previous, word)

    if bigram in bigrams and previous in unigrams:

      # Conditional probability of the word given the previous
      # word. The technical name is *stupid backoff* and it's
      # not a probability distribution but it works well in
      # practice.

      return bigrams[bigram] / total_bigram / self.score(previous)

    # Fall back to using the unigram probability.

    return self.score(word)

  def isegment(self, text):
    "Return iterator of words that is the best segmenation of `text`."
    memo = dict()

    def search(text, previous='<s>'):
      "Return max of candidates matching `text` given `previous` word."
      if text == '':
        return 0.0, []

      def candidates():
        "Generator of (score, words) pairs for all divisions of text."
        for prefix, suffix in self.divide(text):
          prefix_score = math.log10(self.score(prefix, previous))

          pair = (suffix, prefix)
          if pair not in memo:
            memo[pair] = search(suffix, prefix)
          suffix_score, suffix_words = memo[pair]

          yield (prefix_score + suffix_score, [prefix] + suffix_words)

      return max(candidates())

    # Avoid recursion limit issues by dividing text into chunks, segmenting
    # those chunks and combining the results together. Chunks may divide
    # words in the middle so prefix chunks with the last five words of the
    # previous result.

    clean_text = text.upper()
    size = 250
    prefix = ''

    for offset in range(0, len(clean_text), size):
      chunk = clean_text[offset:(offset + size)]
      _, chunk_words = search(prefix + chunk)
      prefix = ''.join(chunk_words[-5:])
      del chunk_words[-5:]
      for word in chunk_words:
        yield word  

    _, prefix_words = search(prefix)

    for word in prefix_words:
      yield word


  def word_segmentation(self, text):
    "Return `text` segmented."
    return ' '.join(list(self.isegment(text)))

  def divide(self, text):
    "Yield `(prefix, suffix)` pairs from `text`."
    for pos in range(1, min(len(text), self.limit) + 1):
        yield (text[:pos], text[pos:])

  def correctSentence(self,sentence, max=5, segmentation=True):
    if segmentation==True:
      sentences_segmented = ' '.join([self.word_segmentation(word) if re.match('^[0-9\+\*\/\(\)\,\=\.\?\¿\:\-]*$', word) is None else word for word in sentence.split()]).lower()
    else:
      sentences_segmented = sentence
    return ' '.join([self.correct(word,max) if re.match('^[0-9\+\*\/\(\)\,\=\.\?\¿\:\-]*$', word) is None else word for word in sentences_segmented.split()])

 

  