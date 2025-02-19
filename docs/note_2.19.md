## Ablation Study
### SmartGD original

![SmartGD Original](<Screenshot 2025-02-18 at 11.35.13.png>)


### SmartGD with Egnn
![Egnn SmartGD](<Screenshot 2025-02-19 at 01.17.23.png>)

## EGNN property

### Modification on original egnn code

[See more details in another file: EGNN](../Egnn/visual.md)

**Conclusion:** Both node embedding h and coordinates x should be equivalent.

### Problems in DeepGD+Egnn

Then edge feature router breaks the eqivariance property. 

Previously I thought this is because edge feature router takes in node feats. But as node feats is equvariant, the edge feature should be equivarant. 

After dismissing edge feature router, the result is as follows:

[See more details in another file: EGNN_DeepGD](../Egnn_DeepGD/visualization/visualization.md)