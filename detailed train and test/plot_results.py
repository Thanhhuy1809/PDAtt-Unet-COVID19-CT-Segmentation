import matplotlib.pyplot as plt
import numpy as np
import os

def parse_results(result_dir):
    """Đọc dữ liệu từ file .txt"""
    txt_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.txt') and f != '3.txt'])
    
    all_metrics = {}
    
    for txt_file in txt_files:
        filepath = os.path.join(result_dir, txt_file)
        run_id = txt_file.replace('.txt', '')
        
        metrics = {
            'train_loss': [], 'valid_loss': [],
            'train_acc': [], 'valid_acc': [],
            'train_dice': [], 'valid_dice': [],
            'train_f1': [], 'valid_f1': [],
            'train_iou': [], 'valid_iou': [],
            'train_sens': [], 'valid_sens': [],
            'train_spec': [], 'valid_spec': [],
        }
        
        with open(filepath, 'r') as f:
            for line in f:
                if 'train Loss:' in line:
                    parts = line.split()
                    metrics['train_loss'].append(float(parts[2]))
                    metrics['train_acc'].append(float(parts[4]))
                    metrics['train_dice'].append(float(parts[6]))
                    metrics['train_f1'].append(float(parts[12]))
                    metrics['train_iou'].append(float(parts[14]))
                    metrics['train_sens'].append(float(parts[18]))
                    metrics['train_spec'].append(float(parts[16]))
                    
                elif 'valid Loss:' in line:
                    parts = line.split()
                    metrics['valid_loss'].append(float(parts[2]))
                    metrics['valid_acc'].append(float(parts[4]))
                    metrics['valid_dice'].append(float(parts[6]))
                    metrics['valid_f1'].append(float(parts[12]))
                    metrics['valid_iou'].append(float(parts[14]))
                    metrics['valid_sens'].append(float(parts[18]))
                    metrics['valid_spec'].append(float(parts[16]))
        
        all_metrics[run_id] = metrics
    
    return all_metrics

def plot_training_curves(metrics, output_dir, run_id='0'):
    """Vẽ biểu đồ huấn luyện"""
    if run_id not in metrics:
        print(f"Run {run_id} not found!")
        return
    
    data = metrics[run_id]
    epochs = range(1, len(data['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Training Results - Run {run_id}', fontsize=14, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(epochs, data['train_loss'], 'b-o', label='Train', markersize=3)
    axes[0, 0].plot(epochs, data['valid_loss'], 'r-s', label='Val', markersize=3)
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, data['train_acc'], 'b-o', label='Train', markersize=3)
    axes[0, 1].plot(epochs, data['valid_acc'], 'r-s', label='Val', markersize=3)
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 2].plot(epochs, data['train_dice'], 'b-o', label='Train', markersize=3)
    axes[0, 2].plot(epochs, data['valid_dice'], 'r-s', label='Val', markersize=3)
    axes[0, 2].set_title('Dice Score', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('Dice')
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    
    # F1-Score
    axes[1, 0].plot(epochs, data['train_f1'], 'b-o', label='Train', linewidth=2, markersize=3)
    axes[1, 0].plot(epochs, data['valid_f1'], 'r-s', label='Val', linewidth=2, markersize=3)
    axes[1, 0].set_title('F1-Score (Primary)', fontweight='bold', color='darkred')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 1].plot(epochs, data['train_iou'], 'b-o', label='Train', markersize=3)
    axes[1, 1].plot(epochs, data['valid_iou'], 'r-s', label='Val', markersize=3)
    axes[1, 1].set_title('IoU', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('IoU')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    
    # Sensitivity vs Specificity
    axes[1, 2].plot(epochs, data['train_sens'], 'g-o', label='Sensitivity', markersize=3)
    axes[1, 2].plot(epochs, data['train_spec'], 'm-s', label='Specificity', markersize=3)
    axes[1, 2].set_title('Sensitivity vs Specificity', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'training_curves_run_{run_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()

if __name__ == '__main__':
    result_dir = './Models3/PYAttUNet/Results'
    
    if not os.path.exists(result_dir):
        print(f"Error: {result_dir} not found!")
        exit()
    
    metrics = parse_results(result_dir)
    
    if not metrics:
        print("No results found!")
        exit()
    
    print(f"Found {len(metrics)} runs")
    print("\nAvailable runs:", list(metrics.keys()))
    
    # Vẽ run 0 làm ví dụ
    if '0' in metrics:
        plot_training_curves(metrics, result_dir, '0')
