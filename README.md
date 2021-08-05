# Json sequence embedding

Given arbitrary JSON input such as:

```json
[
  {
    "title": "hello my name is",
    "subtitle": "another title",
    "metadata": [
      {
        "address": {
          "road": "Corner Frome Road and, North Terrace",
          "city": "Adelaide",
          "state": "SA",
          "postcode": 5000
        }
      }
    ]
  }
]
```

The resnet key-value model preprocesses first into a list of flattened keys and values.

```json
[
  { "title": "hello my name is" },
  { "subtitle": "another title" },
  { "metadata.address.road": "Corner Frome Road and, North Terrace" },
  { "metadata.address.city": "Adelaide" },
  { "metadata.address.state": "SA" },
  { "metadata.address.postcode": "5000" }
]
```

Then treats the whole key as a token, and tokenises each cahracter individually.

```
torch.Tensor(["title", "h", "e", "l", ..., "metadata.address.postcode", "5", "0", "0", "0"])
```

