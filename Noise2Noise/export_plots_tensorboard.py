import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def export_plots(log_dir, output_dir):
    """Export plots from tensorboard log directory to output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()['scalars']
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        # Add a condition to plot the SI-PSNR
        if (tag == 'val/SI-PSNR'):

            print(values)
            
            plt.figure()
            plt.plot(steps, values)
            plt.xlabel('Steps')
            plt.xticks(steps)
            plt.ylim(27, 35)
            plt.yticks(range(27, 36))
            plt.ylabel(tag)
            plt.title("Evolution of the SI-PSNR during validation")
            plt.savefig(os.path.join(output_dir, f"{tag.replace('/', '_')}.png"))
            plt.close()

        else : 
            plt.figure()
            plt.plot(steps, values)
            plt.xlabel('Steps')
            plt.ylabel(tag)
            plt.title(tag)
            plt.savefig(os.path.join(output_dir, f"{tag.replace('/', '_')}.png"))
            plt.close()

if __name__ == "__main__":
    log_dir = "tensorboard_logs"  # Replace with your log directory
    output_dir = "tensorboard_plots"
    export_plots(log_dir, output_dir)