import re
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()

class LogParser:
    def __init__(self):
        self.epochs = []
        self.validation_accs = []
        self.training_accs = []
        self.pattern = re.compile(r"Overall Summary \| Epoch (\d+) \| Train ([0-9.]+) \| Valid ([0-9.]+)")

    def parseLog(self, filename):
        with open(filename, 'r') as file:
            content = file.readlines()
            for line in content:
                match = self.pattern.search(line)
                if match:
                    epoch, train_acc, valid_acc = int(match.group(1)), float(match.group(2)), float(match.group(3))
                    self.epochs.append(epoch)
                    self.validation_accs.append(valid_acc)
                    self.training_accs.append(train_acc)
        # Print statements can be removed if not needed
        #print(self.epochs)
        #print(self.training_accs)
        #print(self.validation_accs)

    def plot_accuracies(self, outname):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_accs, label='Training Accuracy', marker='o')
        plt.plot(self.validation_accs, label='Validation Accuracy', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(fname=outname)

# Usage example
# parser = LogParser()
# parser.parseLog('your_log_file.log')
# parser.plot_accuracies()


if __name__ == '__main__':
    logparser = LogParser()
    p = '/Users/bli/Desktop/dlproject/IDLF23Group4/Visualize/logs/trainer.log'
    logparser.parseLog(p)
    logparser.plot_accuracies('finetune.png')
