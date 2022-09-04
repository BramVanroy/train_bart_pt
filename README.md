As a component of BART, we may do sentence shuffling and so we need sentence splitting. By dfault, we'll use NLTK's 
English punct sentence splitter but by passing a spaCy model name to `spacy_model` (e.g. `en_core_web_sm`) you can 
also rely on spaCy for better (but a lot slower) sentence splitting.

Adapated the transformer defaults to the [given BART args](https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320), specifically:

- poisson_lambda: `3.0` -> `3.5`

