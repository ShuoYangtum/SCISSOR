import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import ViTConfig, ViTForImageClassification, ViTFeatureExtractor, ViTModel, AutoImageProcessor, Swinv2Model, Swinv2ForImageClassification, AutoModel
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import random
import json
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import markov_clustering as mc  
from utils import *
from model import *
random.seed(42)
import argparse


def main():
    parser = argparse.ArgumentParser(description="The hyper-parameters for SCISSOR.")
    parser.add_argument('--train_pth', type=str, help="Your input json file")
    parser.add_argument('--test_pth', type=str, help="Your output json path")
    parser.add_argument('--model_name', type=str, help="The name of model", default="bert-base-uncased")
    parser.add_argument('--model_name', type=str, help="The name of dataset", default="notMNIST")
    parser.add_argument('--tri_num_epochs', type=int, help="The max number of training epochs for the debiasing module.", default=5)
    pparser.add_argument('--cls_num_epochs', type=int, help="The max number of training epochs for the classifier.", default=20)
    
    args = parser.parse_args()
    train_dir=args.train_pth
    test_dir=args.test_pth
    model_name=args.model_name
    dataset_name=args.dataset_name
    tri_num_epochs=args.tri_num_epochs
   cls_num_epochs=args.cls_num_epochs

    down_sampling=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name=='vgg':
        feature_extractor = models.vgg16(pretrained=True)
    elif model_name=='vit':
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224") 
        model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
        model.eval()
    elif model_name=='swin':
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    elif model_name=='dino':
        feature_extractor=AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
    model.eval()
    model=model.to(device)
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
    ])
    
    train_dataset = FilteredImageFolder(train_dir, transform=transform)
    train_dataset = [s for s in train_dataset if s is not None] 
    test_dataset = FilteredImageFolder(test_dir, transform=transform)
    test_dataset = [s for s in test_dataset if s is not None] 
    
    if down_sampling:
       
        samples_per_class = 1000  
       
        class_indices = {}
        for idx, (_, label) in enumerate(train_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        balanced_indices = []
        for label, indices in class_indices.items():
            if len(indices) > samples_per_class:
                sampled_indices = random.sample(indices, samples_per_class)
            else:
                sampled_indices = indices  
            balanced_indices.extend(sampled_indices)
        train_dataset = Subset(train_dataset, balanced_indices)
    
    
    
    images = []
    labels = []
    for image, label in tqdm(train_dataset):
        images.append(image)
        labels.append(label)
        
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    
    cluster_metod='knn'
    
    if cluster_metod=='markov':
        cluster_labels=cluster_images_markov(embeddings, knn_k=num_clusters, inflation=1.4)
    elif cluster_metod=='knn':
        cluster_labels = cluster_images(embeddings, num_clusters)
    
    
    cluster_label_distribution = []
    for cluster_idx in range(len(set(cluster_labels))):
        cluster_data_labels = [int(labels[i]) for i in range(len(labels)) if cluster_labels[i] == cluster_idx]
        label_counts = Counter(cluster_data_labels)
        cluster_label_distribution.append((cluster_idx, label_counts))
    
    cluster_label_distribution.sort(key=lambda x: imbalance_score(x[1]), reverse=True)
    
    
    top_k = 15  
    test_size=100
    
    last_k = len(cluster_label_distribution) - top_k
    
    top_k_imbalanced_clusters = cluster_label_distribution[:top_k]
    top_k_balanced_clusters = cluster_label_distribution[-last_k:]
    
    unb_data = []
    b_data = []
    
    for cluster_idx, label_counts in top_k_imbalanced_clusters:
        for i, img in enumerate(images):
            if cluster_labels[i] == cluster_idx:
                unb_data.append({'image': img, 'label': labels[i], 'cluster': int(cluster_labels[i])})
    
    for cluster_idx, label_counts in top_k_balanced_clusters:
        for i, img in enumerate(images):
            if cluster_labels[i] == cluster_idx:
                b_data.append({'image': img, 'label': labels[i], 'cluster': int(cluster_labels[i])})
    
    
    all_labels = set()
    for _, label_counts in cluster_label_distribution:
        all_labels.update(label_counts.keys())
    all_labels = list(all_labels)
    
    label_counts_imbalanced = {label: sum(cluster[1].get(label, 0) for cluster in top_k_imbalanced_clusters) for label in all_labels}
    label_counts_balanced = {label: sum(cluster[1].get(label, 0) for cluster in top_k_balanced_clusters) for label in all_labels}
    data_size=min([label_counts_balanced[label] for label in all_labels]+[label_counts_imbalanced[label] for label in all_labels])
    
    
    unb_data_samples = {label: random.sample([i for i in unb_data if i['label'] == label], data_size) for label in all_labels}
    b_data_samples = {label: random.sample([i for i in b_data if i['label'] == label], data_size) for label in all_labels}
    
    
    test_unb_data = []
    train_unb_data = []
    test_b_data = []
    train_b_data = []
    
    for label in all_labels:
        random.shuffle(unb_data_samples[label])
        random.shuffle(b_data_samples[label])
        
    
        test_unb_data.extend(unb_data_samples[label][:test_size])
        train_unb_data.extend(unb_data_samples[label][test_size:])
        
        test_b_data.extend(b_data_samples[label][:test_size])
        train_b_data.extend(b_data_samples[label][test_size:])
    
    
    print(f"Train Unbalanced Data Size: {len(train_unb_data)}")
    print(f"Train Balanced Data Size: {len(train_b_data)}")
    print(f"Test Unbalanced Data Size: {len(test_unb_data)}")
    print(f"Test Balanced Data Size: {len(test_b_data)}")
    
    
    ttrain_dataset=train_unb_data
    ttest_dataset=test_unb_data
    tttest_dataset=train_b_data
    
    train_triplets = create_triplets(ttrain_dataset)
    train_loader_tri = DataLoader(TripletDataset(ttrain_dataset, train_triplets), batch_size=32, shuffle=True)
    
    train_loader = DataLoader(ttrain_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(ttest_dataset, batch_size=32, shuffle=False)
    ttest_loader = DataLoader(tttest_dataset, batch_size=32, shuffle=False)
    
    
    NUM_LABELS=2
    
    if model_name=='vgg':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False  
    elif model_name=='vit':
        model = ViT(
            model_name="google/vit-base-patch16-224", 
            num_labels=NUM_LABELS, 
        )
    
        for param in model.parameters():
            param.requires_grad = False  
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.trilinear.parameters():
            param.requires_grad = True
    elif model_name=='swin':
        model = ViT(
            model_name="microsoft/swinv2-tiny-patch4-window8-256", 
            num_labels=NUM_LABELS, 
        )
    
        for param in model.parameters():
            param.requires_grad = False  
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.trilinear.parameters():
            param.requires_grad = True
    elif model_name=='dino':
        model = ViT(
            model_name="facebook/dinov2-base", 
            num_labels=NUM_LABELS, 
        )
    
        for param in model.parameters():
            param.requires_grad = False  
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.trilinear.parameters():
            param.requires_grad = True
            
    model=model.to(device)
    
    criterion = nn.CrossEntropyLoss()  
    criterion_tri=nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    
    for epoch in range(tri_num_epochs):
        model.train()
        running_loss = 0.0
    
        for (triplet_anchors, triplet_positives, triplet_negatives) in tqdm(train_loader_tri):     
    
            optimizer.zero_grad()
    
            triplet_anchors = triplet_anchors.to(device)
            triplet_positives = triplet_positives.to(device)
            triplet_negatives = triplet_negatives.to(device)
    
            anchor_embeds = model(triplet_anchors, skip_classifier=True)
            positive_embeds = model(triplet_positives, skip_classifier=True)
            negative_embeds = model(triplet_negatives, skip_classifier=True)
            loss_tri = criterion_tri(anchor_embeds, positive_embeds, negative_embeds)
    
            loss = loss_tri
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader_tri)}")
    
    
    for param in model.parameters():
        param.requires_grad = False  
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.trilinear.parameters():
        param.requires_grad = False
            
    for epoch in range(cls_num_epochs):
        model.train()
        running_loss = 0.0
        
        for sample in tqdm(train_loader):
            inputs=sample['image']
            labels=sample['label']
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(inputs)
            loss_ce = criterion(outputs, labels)
            loss=loss_ce
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
    
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for sample in ttest_loader:
    
                inputs=sample['image']
                labels=sample['label']
    
                
                inputs, labels = inputs.to(device), labels.to(device) 
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Real-Test Accuracy: {accuracy:.2f}%')
        
        f1 = f1_score(all_labels, all_predictions, average='weighted') 
        precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)
        print(f'Real-Test F1 Score: {f1:.2f}')
        print(precision, recall)


if __name__=="__main__":
    main()