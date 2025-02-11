## Debug DeepGD

### Attempts
1. norm on egnn
2. batch norm on feats
3. batch norm on coordinates
4. node_feats corresponding to graph


### Results
1. Train on 10000 graph, with batch size = 32


| **Approach**                          | **Params**                                      | **Result**                         |
|---------------------------------------|-------------------------------------------------|------------------------------------|
| DeepGD    | same in paper        | 248       |
| SmartGD    | same in paper        | 235       |
| DeepGD+egnn    | bs:32, nf:16, lr:0.001      | 250       |
| DeepGD+egnn    | bs:16, nf:16, lr:0.005      | 252       |
| DeepGD+egnn    | bs:32, nf:8, lr:0.001       | 258       |

## CoRe-GD

### Basic Structure

1. coarsening: go to the coarsest graph
2. Go to GoRe-GD model:
3. Layer Optimization
    - encoding()
    - conv(GRU with message passing, only work in neighbours)
    - decoding(get node pos)
    - rewrite based on pos
    - conv(based on rewrite)
    - repeat from step 2
4. Time Complexity: As long as Stress is used, O(N^2), otherwise less than O(N^2)


### Time on A100
| **Approach**                          | **Time**                    |Size                  | **Result(Stress)**                         |
|---------------------------------------|--------------------------|-----------------------|------------------------------------|
|Zinc dataset | 1 min per epoch |about 23 nodes 50 edge| 8.09|
|Cluster dataset |10 min per epoch|about 117 nodes 4749 edge| 3031|
|Rome dataset| 2 min per epoch|about 20 nodes 50 edge|234.60|

### Thoughts

- As for criteria like stress, large graphs matter.
- How can we exchange information between distant nodes without introducing excessive complexity?

## Note
**Change of the Rome Dataset: https://www.graphdrawing.org/data/rome-graphml.tgz**



# Note 2/5

## SmartGD
**Not work well yet**

| **Approach**                          | **Params**                                      | **Result**                         |
|---------------------------------------|-------------------------------------------------|------------------------------------|
| SmartGD    | params same in paper        | 534 best |

- Not stable enough(GAN is somehow hard to train)


## CoRe-GD
| **Approach**                          | **Time**                                      | **Result**                         |
|---------------------------------------|-------------------------------------------------|------------------------------------|
|Rome dataset| 2 min per epoch |234.60|
| DeepGD    |     0.5 min    | 248       |
| SmartGD    |    1min      | 235       |
| DeepGD+egnn    |   40s   | 250       |
| DeepGD+egnn    | 40s     | 252       |
| DeepGD+egnn    | 40s    | 258       |

## Scalablility

### Time Complexity analysis: 

#### For dense graph, $E = O(V^2)$

Overall complexity can not be lower than $E = O(V^2)$


#### For sparse graph, E = O(V)
1. the rewrite process can be considered a good choice.
   - similar idea: can we use octree to help transfer nearby nodes but with high relative distance?
   - only consider neighbours in a range

2. find a new metric because these metrics are all $O(V^2)$
3. coasening
4. landmark/pivot
5. which node has a minor effect on this node