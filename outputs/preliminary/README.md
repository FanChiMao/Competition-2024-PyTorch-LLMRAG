# Preliminary Submission

## Result

|               | [Submission 1](#submission-1-)<br>(BM25Plus + Embedding) | [Submission 2](#submission-2-)<br>(BM25Plus) | [Submission 3](#submission-3)<br>(Embedding) |
|---------------|:--------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
| Public Score  |                         0.786667                         |                   0.794444                   |                   0.755556                   |
| Private Score |                                                          |                                              |                                              |

<br>

### Submission 1  

> Use BM25+ with embedding reranker (BAAI/bge-small-zh-v1.5) to get the top-3 documents for Kelvin, Jonathan, and Tom,
and use embedding model (BAAI/bge-small-zh-v1.5) to retrieval the top-3 passages for Edward.
Finally, use RRF method to fuse the top-3 documents and top-3 passages to get the final top-1 document.

<details>
<summary>Here is the simple graphics of the flowchart</summary>

```
    
    +-------------------+   +-------------------+   +-------------------+   +--------------------+
    |  Kelvin Pipeline  |   | Jonathan Pipeline |   |   Tom Pipeline    |   |   Edward Pipeline  |
    +-------------------+   +-------------------+   +-------------------+   +--------------------+
                       \               |               /                              |
                        \              |              /                               |
                         \             |             /                                |
                          \            |            /                                 |
                           \           v           /                                  v
                            +---------------------+                        +---------------------+
                            | BM25+ with Reranker |                        | Embedding Retriever |
                            +---------------------+                        +---------------------+
                                       |                                              |
                                       v                                              v
                       +-------------------------------+              +-------------------------------+
                       | Retrieve Top-3 Documents for  |              | Retrieve Top-3 Passages for   |
                       | Kelvin, Jonathan, and Tom     |              | Edward                        |
                       +-------------------------------+              +-------------------------------+
                                                     \                 /
                                                      \               /
                                                       \             /
                                                        \           /
                                                         \         /
                                                          \       /
                                                           \     /
                                                            \   /
                                                             \ /
                                                              |
                                                              v
                                              +-------------------------------+
                                              |      RRF Method for Fusion    |
                                              +-------------------------------+
                                                              |
                                                              v
                                              +-------------------------------+
                                              |     Final Top-1 Document      |
                                              +-------------------------------+
    
```


</details>


<br>

### Submission 2  

> Use BM25+ with embedding reranker (BAAI/bge-reranker-base) to get the top-3 documents for Kelvin, Jonathan, and Tom,
Finally, use RRF method to fuse the top-3 documents to get the final top-1 document.


<details>
<summary>Here is the simple graphics of the flowchart</summary>


```

        +-------------------+   +-------------------+   +-------------------+
        |  Kelvin Pipeline  |   | Jonathan Pipeline |   |   Tom Pipeline    |
        +-------------------+   +-------------------+   +-------------------+
                          \               |               /
                           \              |              /
                            \             |             /
                             \            |            /
                              \           v           /
                               +---------------------+
                               | BM25+ with Reranker |
                               +---------------------+
                                          |
                                          v
                          +-------------------------------+
                          | Retrieve Top-3 Documents for  |
                          | each Pipeline                 |
                          +-------------------------------+
                                          |
                                          v
                          +-------------------------------+
                          |      RRF Method for Fusion    |
                          +-------------------------------+
                                          |
                                          v
                          +-------------------------------+
                          |     Final Top-1 Document      |
                          +-------------------------------+
        
```

</details>

<br>

### Submission 3
> Use embedding model (BAAI/bge-small-zh-v1.5) to retrieval the top-3 passages for Edward.

<details>
<summary>Here is the simple graphics of the flowchart</summary>

```

                                   +--------------------+
                                   |   Edward Pipeline  |
                                   +--------------------+
                                              |
                                              v
                               +-----------------------------+
                               |      Embedding Model        |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |   Retrieve Top-3 Passages   |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |     Final Top-1 Results     |
                               +-----------------------------+

```

</details>

<br><br><br><br>