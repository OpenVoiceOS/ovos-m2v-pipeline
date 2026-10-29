# Text embeddings

The package also ships a text embeddings plugin, `ovos-m2v-embeddings-plugin`. It turns text into vectors with a Model2Vec static model. Retrieval, deduplication and clustering use these vectors. The plugin needs no GPU and no network once the model is in the Hugging Face cache.

## Load the plugin

The plugin registers under the `opm.embeddings.text` entry point:

```python
from ovos_plugin_manager.embeddings import load_text_embeddings_plugin

embedder = load_text_embeddings_plugin("ovos-m2v-embeddings-plugin")({"model": "minishlab/potion-multilingual-128M"})
vectors = embedder.get_embeddings_batch(["Qual é a capital de Portugal?", "Lisboa é a capital."])
similarity = float(vectors[0] @ vectors[1])
```

`get_embeddings(text)` returns one vector. `get_embeddings_batch(texts)` returns one vector per text, in the order of `texts`, from one `encode` call. Use the batch method for many texts: for 5000 passages it is about 4 times faster than one call per text.

## Configuration

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model` | `str` | `minishlab/potion-multilingual-128M` | Hugging Face repo ID or local path of a Model2Vec model. |
| `normalize` | `bool` | `true` | Scale each vector to length 1, so the dot product of two vectors is their cosine similarity. |

## Share one model with the intent pipeline

The plugin loads its model through the same process-wide cache as the intent pipelines. When the embedder and a pipeline load the same path, the process holds one copy of the model. The pipelines resolve a Hub id to a folder of the local Hugging Face cache before they load it, so the copy is shared only when both name the same local directory.

An intent model, such as `OpenVoiceOS/ovos-m2v-intents-multilingual`, is trained to separate intents. For retrieval over documents, use a general model such as `minishlab/potion-multilingual-128M`.
