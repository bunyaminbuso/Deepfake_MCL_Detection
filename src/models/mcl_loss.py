import torch
import torch.nn as nn

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self):
        super(SupervisedContrastiveLoss, self).__init__()

    def forward(self, v_embed, a_embed, labels):
        # Cosine Benzerliği (-1.0 ile +1.0)
        similarity = torch.sum(v_embed * a_embed, dim=1)
        
        # Gercek (0): Benzerligi +1'e yaklastir -> Loss = 1 - similarity
        # Sahte  (1): Benzerligi -1'e yaklastir -> Loss = 1 + similarity
        loss_real = (1.0 - similarity) * (1.0 - labels.float())
        loss_fake = (1.0 + similarity) * labels.float()
        
        total_loss = torch.mean(loss_real + loss_fake)
        return total_loss