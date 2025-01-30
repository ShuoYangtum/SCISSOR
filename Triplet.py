def create_triplets(dataset):
    triplets = []
    
    grouped_by_label = {}
    for idx, data in enumerate(dataset):
        label = int(data['label'])
        cluster = data['cluster']
        if label not in grouped_by_label:
            grouped_by_label[label] = []
        grouped_by_label[label].append((idx, cluster))
    
    for label, samples in tqdm(grouped_by_label.items()):
        for i, (anchor_idx, anchor_cluster) in enumerate(samples):

            positive_samples_1 = [s[0] for s in samples if s[1] != anchor_cluster]
            negative_samples_1 = [s[0] for l, s_list in grouped_by_label.items() if l != label for s in s_list]
            
            if positive_samples_1 and negative_samples_1:
                positive_idx = random.choice(positive_samples_1) 
                negative_idx = random.choice(negative_samples_1) 
                triplets.append((anchor_idx, positive_idx, negative_idx))

            positive_samples_2 = positive_samples_1  
            negative_samples_2 = [s[0] for s in samples if s[1] == anchor_cluster and s[0] != anchor_idx]
            
            if positive_samples_2 and negative_samples_2:
                positive_idx = random.choice(positive_samples_2)
                negative_idx = random.choice(negative_samples_2)
                triplets.append((anchor_idx, positive_idx, negative_idx))

            positive_samples_3 = negative_samples_2
            negative_samples_3 = negative_samples_1
            
            if positive_samples_3 and negative_samples_3:
                positive_idx = random.choice(positive_samples_3)
                negative_idx = random.choice(negative_samples_3)  
                triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, triplets):
        self.dataset = dataset
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        anchor = self.dataset[anchor_idx]['image']
        positive = self.dataset[positive_idx]['image']
        negative = self.dataset[negative_idx]['image']
        return anchor, positive, negative

class CustomQuadrupletLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=1.5):
        super(CustomQuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.relu = nn.ReLU()
    
    def forward(self, anchor, positive, negative1, negative2):

        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg1_dist = torch.norm(anchor - negative1, p=2, dim=1)
        neg2_dist = torch.norm(anchor - negative2, p=2, dim=1)
        
        loss_pos = pos_dist.mean()

        loss_neg1 = self.relu(pos_dist - neg1_dist + self.margin1).mean()

        loss_neg2 = self.relu(neg1_dist - neg2_dist + self.margin2).mean()
        loss_neg2 += self.relu(pos_dist - neg2_dist + self.margin2).mean()

        total_loss = loss_pos + loss_neg1 + loss_neg2
        return total_loss