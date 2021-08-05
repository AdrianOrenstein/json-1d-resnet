# Json sequence embedding

Given a JSON input such as:

```json
{
  "input": {
    "title": "hello my name is",
    "subtitle": "another title",
    "address": {
      "road": "Corner Frome Road and, North Terrace",
      "city": "Adelaide",
      "state": "SA",
      "postcode": 5000
    }
  },
  "label": 1
}
```

To preprocess the input, the resnet key-value model preprocesses first into a list of flattened keys and values.

```json
[
  { "title": "hello my name is" },
  { "subtitle": "another title" },
  { "address.road": "Corner Frome Road and, North Terrace" },
  { "address.city": "Adelaide" },
  { "address.state": "SA" },
  { "address.postcode": "5000" }
]
```

Then treats the whole key as a token, and tokenises each character individually.

```python
tokens = ["title", "h", "e", "l", ..., "address.postcode", "5", "0", "0", "0"]

embedded_sequence: torch.LongTensor = tokeniser.convert_tokens_to_ids(tokens)

logits = key_value_resnet(embedded_sequence)
```

## Pros

- The model's vocabulary is setup to use all keys in the vocabulary + all [printable runes from string.printable](https://docs.python.org/3/library/string.html#string.printable), this is a combination of digits, ascii_letters, punctuation, and whitespace.

## Cons

- Keys needs to have been added to the vocabulary of the model, any unknown keys will be assigned the "UNK" token

## TODO

- [ ] MVP

  - [x] start readme
  - [ ] an example with homemade json dataset
  - [ ] make a `run_experiments/local_dataset_bench.sh`

- [ ] Extra

  - [ ] look into [pytorch-lightning-transformers](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/text_classification.html)?
  - [ ] bench [Yahoo\! Answers](https://paperswithcode.com/sota/text-classification-on-yahoo-answers) because it has fast-text for comparison
  - [ ] bench [IMDB sequence classification problem](https://paperswithcode.com/sota/text-classification-on-imdb) make a `run_experiments/imdb.sh` and post results
