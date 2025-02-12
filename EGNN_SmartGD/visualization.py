import matplotlib.pyplot as plt
import networkx as nx
import os
def visualization_batch(batch, pos, epoch=0, batch_idx=0, all = False, method = None):
    pos = pos.detach().cpu()
    count = 0
    save_dir = 'visualization'
    for i in range(len(batch)):
        plt.clf()
        pos_dict = {i-count: (pos[i, 0].item(), pos[i, 1].item()) for i in range(pos.shape[0])}
        count += batch[i].G.number_of_nodes()
        nx.draw(batch[i].G, pos=pos_dict, with_labels=True)
        os.makedirs(save_dir, exist_ok=True)
        if method:
            save_dir = os.path.join(save_dir, method)
            os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, f"graph_epoch_{epoch}_batch_{batch_idx}_graph_{i}.png")
        plt.savefig(image_path)
        print(f"Graph saved to {image_path}")
        if all == False:
            break