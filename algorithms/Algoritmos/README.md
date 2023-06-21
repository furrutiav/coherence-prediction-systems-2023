# KidESpell

A python Spanish spell checking oriented to child, based on [KidSpell](https://aclanthology.org/2020.lrec-1.857/)

## Install

To install from source:

```
git clone https://github.com/maltamiranomontero/KidESpell.git
cd KidESpell 
python setup.py install
```

## Quickstart

To use it:

```python
from kidespell import SpellChecker

SC = SpellChecker()

SC.getSuggestionsForSentence('ce stas asiendo')
#Return: {'asiendo': ['haciendo', 'siendo', 'asiento', 'habiendo', 'hacienda'],
# 'ce': ['que', 'ce', 'cae', 'co', 'cc'],
# 'stas': ['estas', 'seas', 'ptas', 'citas']}

```
