---
language: de
license: ???
---

# electra-base-german-uncased

## How to use
**The usage description above - provided by Hugging Face - is wrong! Please use this:**

### Transformers Usage
```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("model_name", strip_accents=False)
model = AutoModelWithLMHead.from_pretrained("model_name")
```

### FARM Usage
```python
# TODO
```

## Case and Umlauts
This model is uncased. We are convinced that capitalization does not add semantic value in German language.
Quite the opposite is the case. Many words are written in upper case at the beginning of the sentence and in
lower case in the sentence. The model would have to learn that these are the same words.

However, the German umlauts do make a semantic difference. Therefore this model does not do without umlauts.
The necessary parameter is `strip_accents=False` and needs to be set for the tokenizer.
It was added to Transformers with [PR #6280](https://github.com/huggingface/transformers/pull/6280).

Since Transformers has not been released since the PR was merged you have to install the development
branch: `pip install git+https://github.com/huggingface/transformers.git -U`

## Creators
This model was trained and open sourced in equal parts by:
- [Philip May](https://eniak.de) - [T-Systems on site services GmbH](https://www.t-systems-onsite.de/)
- Philipp Reißel - [ambeRoad](https://amberoad.de/)

## Pre-training details

## Performance on downstream tasks
