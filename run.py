import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import os
from model import MobileViT
from modules import *
import time
import argparse

path = 'image_enhance/mixup_cifar100_baseline'

def get_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def train_model(resume_checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = MobileViT2().to(device)
    model = MobileViT().to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.002,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=25,
        eta_min=0.0002
    )
    
    scaler = GradScaler()
    
    trainloader, testloader = get_loaders()
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    best_acc = 0.0
    best_epoch = 0
    start_epoch = 0
    
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed training from epoch {start_epoch} with best accuracy {best_acc:.2f}%")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, 200):
        epoch_start_time = time.time()
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels).item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            mixed_inputs, _, _, _ = mixup_data(inputs, labels)
            with autocast('cuda'):
                _ = model(mixed_inputs)
            
            running_loss += loss.item() * batch_size
            total += batch_size
        
        scheduler.step()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * running_corrects / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training Acc: {epoch_acc:.2f}%')
        
        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        test_losses.append(test_loss / len(testloader))
        test_accuracies.append(100. * correct / total)
        
        print(f'Test Loss: {test_loss / len(testloader):.4f}')
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        
        if 100. * correct / total > best_acc:
            best_acc = 100. * correct / total
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'output/{path}/best_model.pth')
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch+1} duration: {epoch_duration:.2f} seconds')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc
        }
        torch.save(checkpoint, f'output/{path}/checkpoint.pth')
    
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')
    
    
    epochs = range(1, 201)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'output/{path}/loss_acc.png')
    plt.close()
    
    with open(f'output/{path}/best_result.txt', 'w') as f:
        f.write(f'Best Test Accuracy: {best_acc:.2f}%\n')
        f.write(f'Best Epoch: {best_epoch}\n')
        f.write('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileViT model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists(f'output/{path}'):
        os.makedirs(f'output/{path}')
    
    train_model(resume_checkpoint=args.resume)
