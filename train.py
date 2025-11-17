"""
Training Script for Sign Language Recognition Model
Trains the multi-stream TCN-xLSTM model on sign language data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np

# Add models directory to path
sys.path.append('models')

from models.sign_language_model import create_sign_language_model
from dataset import create_dataloaders


class Trainer:
    """
    Trainer class for sign language recognition model
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler=None,
                 device='cuda',
                 save_dir='checkpoints',
                 log_dir='logs',
                 num_classes=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training stats
        self.epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.num_classes = num_classes
        
    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]")
        for batch_idx, (body, hand, face, labels) in enumerate(pbar):
            # Move data to device
            body = body.to(self.device)
            hand = hand.to(self.device)
            face = face.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(body, hand, face)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class accuracy
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]")
            for body, hand, face, labels in pbar:
                # Move data to device
                body = body.to(self.device)
                hand = hand.to(self.device)
                face = face.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(body, hand, face)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Per-class statistics
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / len(self.val_loader),
                    'acc': 100. * correct / total
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Compute per-class accuracy
        class_accuracies = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_accuracies[i] = 100. * class_correct[i] / class_total[i]
        
        return avg_loss, accuracy, class_accuracies
    
    def train(self, num_epochs):
        """
        Train the model for multiple epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, class_accs = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, self.epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, self.epoch)
            if self.scheduler:
                self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)
            
            # Print epoch summary
            print(f"\nEpoch {self.epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_acc.pth', val_acc=val_acc, val_loss=val_loss)
                print(f"  ✓ Saved best accuracy model: {val_acc:.2f}%")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss.pth', val_acc=val_acc, val_loss=val_loss)
                print(f"  ✓ Saved best loss model: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if self.epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth', val_acc=val_acc, val_loss=val_loss)
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.writer.close()
    
    def save_checkpoint(self, filename, **kwargs):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, filename):
        """
        Load model checkpoint
        """
        load_path = self.save_dir / filename
        if not load_path.exists():
            print(f"Checkpoint not found: {load_path}")
            return
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Sequence length (number of frames)')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of sign language classes')
    
    # Model parameters
    parser.add_argument('--xlstm_type', type=str, default='mlstm', choices=['mlstm', 'slstm'],
                        help='Type of xLSTM to use')
    parser.add_argument('--tcn_hidden_dim', type=int, default=256,
                        help='Hidden dimension for TCN')
    parser.add_argument('--xlstm_hidden_dim', type=int, default=256,
                        help='Hidden dimension for xLSTM')
    parser.add_argument('--xlstm_num_layers', type=int, default=2,
                        help='Number of xLSTM layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = create_sign_language_model(
        model_type='classifier',
        num_classes=args.num_classes,
        xlstm_type=args.xlstm_type,
        tcn_hidden_dim=args.tcn_hidden_dim,
        xlstm_hidden_dim=args.xlstm_hidden_dim,
        xlstm_num_layers=args.xlstm_num_layers
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        num_classes=args.num_classes
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(args.num_epochs)
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
